import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
import numpy as np
from scripts import evaluate_label_real_anomalies as ev


class BottomSelectionTest(unittest.TestCase):
    def test_fraction_argument_default_and_override(self):
        with patch.object(sys,'argv',['eval','run','--top-fraction','.02']):
            self.assertIsNone(ev.parse_args().bottom_fraction)
        with patch.object(sys,'argv',['eval','run','--bottom-fraction','.1']):
            self.assertEqual(ev.parse_args().bottom_fraction,.1)

    def test_global_bottom_order_and_post_cut_veto(self):
        for veto_all in (False,True):
            with self.subTest(veto_all=veto_all), tempfile.TemporaryDirectory() as temp:
                selector=ev.BottomEventSelection(Path(temp),.5,8,2,3)
                scores=np.array([8,3,6,1,5,2,7,4],dtype=np.float32)
                events=np.zeros((8,2,3),dtype=np.float32)
                events[:,0,0]=.4
                events[:,0,2]=scores  # Track correspondence between saved events and scores.
                events[scores==2,0,1]=1
                if veto_all: events[scores<=4,0,1]=1
                for start in range(0,8,3):
                    selector.update(events[start:start+3],scores[start:start+3])
                result=selector.finish(['part_pt','part_isMuon','identifier'])
                self.assertEqual(result['cutoff'],4)
                self.assertEqual(result['bottom_count_before_muon_veto'],4)
                expected=[] if veto_all else [1,3,4]
                np.testing.assert_array_equal(np.load(result['event_scores_path']),expected)
                np.testing.assert_array_equal(np.load(result['events_path'])[:,0,2],expected)
                self.assertEqual(result['bottom_count'],len(expected))


if __name__=='__main__': unittest.main()
