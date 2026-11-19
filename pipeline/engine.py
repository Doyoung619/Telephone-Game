"""Batched local editor, content-addressed cache and durable state journal."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference'))
from local_editor import make_edit_prompt, COMMON


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


class Journal:
    def __init__(self, path):
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, body TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS states (key TEXT PRIMARY KEY, phase TEXT, source TEXT, body TEXT)')
        self.db.execute('CREATE INDEX IF NOT EXISTS phase_idx ON states(phase)')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, body TEXT)')
        self.db.commit()

    def metadata(self, key):
        row = self.db.execute('SELECT body FROM metadata WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_metadata(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (key, json.dumps(value)))

    def cached(self, key):
        row = self.db.execute('SELECT body FROM cache WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_cache(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO cache VALUES (?,?)', (key, json.dumps(value)))

    def save(self, state):
        self.db.execute('INSERT OR REPLACE INTO states VALUES (?,?,?,?)',
                        (state['state_id'], state['phase'], state['source_id'], json.dumps(state)))

    def rows(self, phase=None):
        cur = self.db.execute('SELECT body FROM states' + (' WHERE phase=?' if phase else ''),
                              (phase,) if phase else ())
        return [json.loads(row[0]) for row in cur]

    def get(self, key):
        row = self.db.execute('SELECT body FROM states WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def commit(self):
        self.db.commit()


class BatchedEditor:
    def __init__(self, journal, batch_size=32, mock=False):
        self.journal = journal
        self.batch_size = batch_size
        self.mock = mock
        self.counters = dict(batch_invocations=0, input_sequences=0, cache_hits=0,
                             input_tokens=0, output_tokens=0, inference_seconds=0., oom_retries=0)
        self.counters.update(journal.metadata('editor_counters') or {})
        if mock:
            self.metadata = dict(model_id='MOCK_FUNCTIONAL_TEST_ONLY', resolved_revision='none',
                                 decoding={'do_sample': False, 'num_beams': 1, 'max_new_tokens': 256})
            return
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        torch.set_num_threads(8)
        lock = json.loads((ROOT / 'model_lock.json').read_text())
        start = time.perf_counter()
        opts = dict(revision=lock['revision'], local_files_only=True, trust_remote_code=False)
        self.tokenizer = AutoTokenizer.from_pretrained(lock['model_id'], padding_side='left', **opts)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not torch.cuda.is_available():
            raise RuntimeError('This frozen Qwen GPU profile requires CUDA; CPU fallback is a separate study.')
        self.model = AutoModelForCausalLM.from_pretrained(lock['model_id'], torch_dtype=torch.float16,
                                                       attn_implementation='sdpa', **opts).to('cuda').eval()
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.metadata = dict(model_id=lock['model_id'], resolved_revision=lock['revision'],
            tokenizer_revision=lock['revision'], torch=torch.__version__, transformers=transformers.__version__,
            device=torch.cuda.get_device_name(), dtype='float16', attention='sdpa',
            batch_size=batch_size, input_token_cap=768,
            decoding={'do_sample': False, 'num_beams': 1, 'max_new_tokens': 256},
            prompt_source_hash=digest((ROOT/'reference/local_editor.py').read_text()),
            load_seconds=time.perf_counter()-start)
        print(json.dumps({'event':'model_loaded', **self.metadata}), flush=True)

    def prompt(self, text, role, max_words, variant='main'):
        if role == 'R':
            return COMMON + '\n\nTASK: Relay the document faithfully, preserving its meaning and format.\n\n<document>\n' + text + '\n</document>'
        p = make_edit_prompt(text, role, max_words)
        if variant == 'three_bullets' and role == 'D':
            p = p.replace('one to eight nonempty bullet lines', 'exactly three nonempty bullet lines')
        return p

    def edit_tasks(self, tasks):
        """Only whitelisted current text / role / absolute limit enter the editor."""
        outputs = [None] * len(tasks)
        pending = {}
        for i, t in enumerate(tasks):
            prompt = self.prompt(t['text'], t['role'], t['max_words'], t.get('variant', 'main'))
            identity = {k:v for k,v in self.metadata.items() if k != 'load_seconds'}
            key = digest({'prompt':prompt, 'model':identity})
            cached = self.journal.cached(key)
            if cached is not None:
                outputs[i] = {**cached, 'cache_hit':True, 'cache_key':key}
                self.counters['cache_hits'] += 1
            else:
                pending.setdefault(key, {'prompt':prompt, 'indices':[]})['indices'].append(i)
        items = list(pending.items())
        for offset in range(0, len(items), self.batch_size):
            chunk = items[offset:offset+self.batch_size]
            results = self._generate([x[1]['prompt'] for x in chunk])
            for (key, item), result in zip(chunk, results):
                result['prompt_hash'] = digest(item['prompt'])
                if result['status'] == 'ok':
                    self.journal.save_cache(key, result)
                for pos, i in enumerate(item['indices']):
                    outputs[i] = {**result, 'cache_hit':pos > 0, 'cache_key':key}
                    self.counters['cache_hits'] += int(pos > 0)
            self.journal.save_metadata('editor_counters', self.counters)
            self.journal.commit()
        self.journal.save_metadata('editor_counters', self.counters)
        self.journal.commit()
        return outputs

    def _generate(self, prompts):
        if self.mock:
            self.counters['input_sequences'] += len(prompts)
            self.counters['batch_invocations'] += 1
            return [dict(text=p.split('<document>\n')[1].split('\n</document>')[0], input_tokens=0,
                         output_tokens=0, seconds=0., batch_seconds=0., hit_token_cap=False, status='ok') for p in prompts]
        texts = [self.tokenizer.apply_chat_template([{'role':'user','content':p}], tokenize=False,
                                                   add_generation_prompt=True) for p in prompts]
        tokenized = self.tokenizer(texts, padding=True, return_tensors='pt', truncation=False)
        lengths = tokenized['attention_mask'].sum(dim=1).tolist()
        oversized = [i for i, n in enumerate(lengths) if n > 768]
        if oversized:
            results = [None] * len(prompts)
            keep = [i for i in range(len(prompts)) if i not in oversized]
            for i in oversized:
                results[i] = dict(text='', input_tokens=lengths[i], output_tokens=0, seconds=0.,
                    hit_token_cap=False, status='input_cap_exceeded', failure=True)
            if keep:
                for i, result in zip(keep, self._generate([prompts[i] for i in keep])):
                    results[i] = result
            return results
        inputs = tokenized.to('cuda')
        self.torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=256,
                    use_cache=True, pad_token_id=self.tokenizer.pad_token_id)
            self.torch.cuda.synchronize()
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            if len(prompts) == 1:
                raise
            self.counters['oom_retries'] += 1
            half = len(prompts)//2
            return self._generate(prompts[:half]) + self._generate(prompts[half:])
        elapsed = time.perf_counter() - start
        outputs = generated[:,inputs['input_ids'].shape[1]:].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos,list) else [eos])
        results = []
        for nin, tokens in zip(lengths, outputs):
            end = next((i+1 for i,t in enumerate(tokens) if t in eos), len(tokens))
            tokens = tokens[:end]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            cap = len(tokens) >= 256 and (not tokens or tokens[-1] not in eos)
            status = 'output_cap_hit' if cap else ('ok' if text else 'empty_output')
            results.append(dict(text=text, input_tokens=nin, output_tokens=len(tokens),
                seconds=elapsed/len(prompts), batch_seconds=elapsed, batch_size=len(prompts),
                hit_token_cap=cap, ended_with_eos=bool(tokens and tokens[-1] in eos), status=status))
            self.counters['input_tokens'] += nin
            self.counters['output_tokens'] += len(tokens)
        self.counters['input_sequences'] += len(prompts)
        self.counters['batch_invocations'] += 1
        self.counters['inference_seconds'] += elapsed
        return results
