"""Offline functional checks. No model loading, downloads or GPU experiments."""
from collections import Counter
import itertools
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pipeline'))

import analysis
from data_eval import prepare_data, read_jsonl, evaluate
from engine import BatchedEditor, Journal
from prepare import register_external_refit
from refit_external import fit_a1
from run_all import Study


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temporary.name)
        cls.data=cls.root/'data'
        prepare_data(cls.data,profile='paper',seed=11)
        cls.public=read_jsonl(cls.data/'public_documents.jsonl')
        cls.private={r['source_id']:r for r in read_jsonl(cls.data/'private_gold.jsonl')}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_group_splits_and_witness_feasibility(self):
        self.assertEqual(Counter(r['split'] for r in self.public),
            {'calibration_fit':64,'calibration_safety':16,'selection':40,'test':80})
        self.assertEqual(len({r['group_id'] for r in self.public}),200)
        for record in self.public:
            gold=self.private[record['source_id']]
            self.assertLessEqual(len(gold['feasibility_witness'].split()),record['requirements']['max_words'])
            self.assertEqual(evaluate(record['source_text'],record,gold)['F'],1)
            self.assertEqual(evaluate(gold['feasibility_witness'],record,gold)['S'],1)

    def test_all_prefixes_and_durable_resume_without_model_calls(self):
        path=self.root/'resume'
        first=Study(path,mock=True,data_dir=self.data)
        with first.stage('fixture_bank',64):
            rows=first.bank(self.public[:1],self.private,'fixture_bank')
        self.assertEqual(len(rows),65)
        terminals=[r for r in rows if r['hop']==4]
        self.assertEqual({tuple(r['prefix']) for r in terminals},set(itertools.permutations(range(4))))
        counters=dict(first.editor.counters)
        timing=dict(first.timings['fixture_bank'])
        first.journal.db.close()
        second=Study(path,mock=True,data_dir=self.data)
        self.assertEqual(second.editor.counters,counters)
        with second.stage('fixture_bank',64):
            resumed=second.bank(self.public[:1],self.private,'fixture_bank')
        self.assertEqual(resumed,rows)
        self.assertEqual(second.editor.counters,counters)
        self.assertEqual(second.timings['fixture_bank']['input_sequences'],timing['input_sequences'])
        self.assertGreaterEqual(second.timings['fixture_bank']['wall_seconds'],timing['wall_seconds'])
        second.journal.db.close()

    def test_prompt_whitelist_and_fixed_absolute_word_cap(self):
        journal=Journal(self.root/'prompts.sqlite')
        editor=BatchedEditor(journal,mock=True)
        record=self.public[0];cap=record['requirements']['max_words']
        record={**record,'private_gold':'DO_NOT_LEAK_PRIVATE','evaluation_queries':'DO_NOT_LEAK_QUERIES'}
        for text in [record['source_text'],'A much shorter current document.']:
            prompt=editor.prompt(text,'B',cap)
            self.assertIn(f'at most {cap} English words',prompt)
            self.assertNotIn('DO_NOT_LEAK_PRIVATE',prompt)
            self.assertNotIn('DO_NOT_LEAK_QUERIES',prompt)
        self.assertIn('exactly three',editor.prompt('Text','D',cap,'three_bullets'))
        journal.db.close()
        study=Study(self.root/'privacy',mock=True,data_dir=self.data)
        captured=[];original=study.editor.edit_tasks
        def spy(tasks):
            captured.extend(tasks)
            return original(tasks)
        study.editor.edit_tasks=spy
        with study.stage('privacy_bank',64):
            study.bank([record],self.private,'privacy_bank')
        payload=json.dumps(captured)
        self.assertNotIn('DO_NOT_LEAK_PRIVATE',payload)
        self.assertNotIn('DO_NOT_LEAK_QUERIES',payload)
        self.assertTrue(all(set(task)=={'text','role','max_words','variant'} for task in captured))
        study.journal.db.close()

    def test_selection_fitting_rejects_test_records(self):
        with self.assertRaisesRegex(ValueError,'calibration'):
            analysis.fit_candidates([next(r for r in self.public if r['split']=='test')],[])

    def test_refit_requires_complete_fit_only_bank_and_excludes_null_sources(self):
        orders=list(itertools.permutations(range(4)));target=orders[7]
        rows=[{'source_id':sid,'prefix':order,'F':None if sid=='unscorable' else float(order==target)}
              for sid in ['fit','unscorable'] for order in orders]
        result=fit_a1(rows,['fit','unscorable'])
        self.assertEqual(tuple(result['order']),target)
        self.assertEqual(result['excluded_fit_source_ids'],['unscorable'])
        with self.assertRaisesRegex(ValueError,'Non-fit'):
            fit_a1(rows+[{'source_id':'test','prefix':orders[0],'F':1}],['fit','unscorable'])
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            fit_a1(rows[1:],['fit','unscorable'])

    def test_preregistration_cannot_be_created_after_external_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data/external';data.mkdir(parents=True)
            records=[{'source_id':f'fit_{i}','dataset':'example','split':'calibration_fit'} for i in range(16)]
            (data/'public_documents.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            run=root/'runs/main';run.mkdir(parents=True)
            with sqlite3.connect(run/'journal.sqlite') as db:
                db.execute('CREATE TABLE states (phase TEXT)')
                db.execute("INSERT INTO states VALUES ('E1_external_example')")
            with self.assertRaisesRegex(RuntimeError,'retrospective'):
                register_external_refit(root)
            self.assertFalse((run/'external/refit_preregistration.json').exists())


if __name__=='__main__':
    unittest.main()
