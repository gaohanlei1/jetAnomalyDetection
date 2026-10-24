import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from scripts import evaluate_label_real_anomalies as ev


class LeadingMuonVetoTest(unittest.TestCase):
    def test_leading_particle_threshold_and_identity(self):
        # Features deliberately reordered; include padding and a subleading muon.
        x=np.zeros((5,3,2),dtype=np.float32)
        x[0,:2]=[[1,.31],[0,.2]]
        x[1,:2]=[[1,.30],[0,.2]]
        x[2,:2]=[[1,.31],[0,.5]]
        x[3,:2]=[[0,.8],[1,.1]]
        np.testing.assert_array_equal(ev.leading_muon_veto(x,['part_isMuon','part_pt']),
                                     [True,False,False,False,False])

    def test_post_cut_no_backfill_and_empty_output(self):
        for all_veto in (False,True):
            with self.subTest(all_veto=all_veto), tempfile.TemporaryDirectory() as temp:
                x=torch.zeros(4,2,2)
                x[:,0,0]=.4
                x[3,0,1]=1
                if all_veto: x[2,0,1]=1
                batch={'x_particles':x,'padding_mask':x.eq(0).all(-1)}
                with patch.object(ev,'encode_single_view',return_value=torch.ones(4,2)), \
                     patch.object(ev,'knn_scores',return_value=np.array([1,2,3,4],dtype=np.float32)):
                    result=ev.score_and_select_top_events(model=None,loader=[batch],total_events=4,
                        reference=torch.ones(2,2),k=1,reduction='mean',reference_batch_size=2,
                        top_fraction=.5,device=torch.device('cpu'),precision='fp32',max_steps=None,
                        output_dir=Path(temp),max_num_particles=2,num_features=2,
                        feature_names=['part_pt','part_isMuon'],heap_progress_every=100)
                self.assertEqual(result['cutoff'],3)
                self.assertEqual(result['top_count_before_muon_veto'],2)
                self.assertEqual(result['top_count'],0 if all_veto else 1)
                scores=np.load(result['event_scores_path'])
                np.testing.assert_array_equal(scores,[] if all_veto else [3])
                self.assertEqual(np.load(result['events_path']).shape,(len(scores),2,2))
                self.assertEqual(len(np.load(result['all_scores_path'])),4)


if __name__=='__main__': unittest.main()
