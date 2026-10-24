import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from scripts import evaluate_label_real_anomalies as ev


class LeadingLeptonVetoTest(unittest.TestCase):
    def test_leading_particle_threshold_and_identity(self):
        # Reordered features: muon, pt fraction, electron. Include exact boundary,
        # subleading leptons, non-leptons, and an all-padding event.
        x=np.zeros((7,3,3),dtype=np.float32)
        x[0,:2]=[[1,.21,0],[0,.1,0]]
        x[1,:2]=[[1,.20,0],[0,.1,0]]
        x[2,:2]=[[0,.21,1],[0,.1,0]]
        x[3,:2]=[[0,.20,1],[0,.1,0]]
        x[4,:2]=[[1,.31,0],[0,.5,0]]
        x[5,:2]=[[0,.8,0],[0,.3,1]]
        np.testing.assert_array_equal(ev.leading_lepton_veto(x,['part_isMuon','part_pt','part_isElectron']),
                                     [True,False,True,False,False,False,False])

    def test_post_cut_no_backfill_and_empty_output(self):
        for all_veto in (False,True):
            with self.subTest(all_veto=all_veto), tempfile.TemporaryDirectory() as temp:
                x=torch.zeros(4,2,3)
                x[:,0,0]=.4
                x[3,0,1]=1
                if all_veto: x[2,0,1]=1
                batch={'x_particles':x,'padding_mask':x.eq(0).all(-1)}
                with patch.object(ev,'encode_single_view',return_value=torch.ones(4,2)), \
                     patch.object(ev,'knn_scores',return_value=np.array([1,2,3,4],dtype=np.float32)):
                    result=ev.score_and_select_top_events(model=None,loader=[batch],total_events=4,
                        reference=torch.ones(2,2),k=1,reduction='mean',reference_batch_size=2,
                        top_fraction=.5,device=torch.device('cpu'),precision='fp32',max_steps=None,
                        output_dir=Path(temp),max_num_particles=2,num_features=3,
                        feature_names=['part_pt','part_isMuon','part_isElectron'],heap_progress_every=100)
                self.assertEqual(result['cutoff'],3)
                self.assertEqual(result['top_count_before_lepton_veto'],2)
                self.assertEqual(result['top_count'],0 if all_veto else 1)
                scores=np.load(result['event_scores_path'])
                np.testing.assert_array_equal(scores,[] if all_veto else [3])
                self.assertEqual(np.load(result['events_path']).shape,(len(scores),2,3))
                self.assertEqual(len(np.load(result['all_scores_path'])),4)
                np.testing.assert_array_equal(np.load(result['vetoed_event_scores_path']),
                                              [4,3] if all_veto else [4])
                self.assertEqual(result['leading_lepton_veto_count'],2 if all_veto else 1)

    def test_four_group_plots(self):
        from datasets import cms_streaming as cms
        features=cms.PARTICLE_FEATURES
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            x=np.zeros((4,2,len(features)),dtype=np.float32)
            x[:,0,features.index('part_pt')]=.3
            x[:,0,features.index('part_isElectron')]=1
            x[:,0,features.index('part_deta')]=.1
            x[:,0,features.index('part_charge')]=-1
            for tail in ('top','bottom'):
                selection={f'{tail}_count':4, 'leading_lepton_veto_count':4}
                scores=np.array([4,3,2,1] if tail=='top' else [1,2,3,4],dtype=np.float32)
                for prefix in ('','vetoed_'):
                    event_path=root/f'{tail}_{prefix}events.npy'
                    score_path=root/f'{tail}_{prefix}event_scores.npy'
                    np.save(event_path,x);np.save(score_path,scores)
                    selection[prefix+'events_path']=event_path
                    selection[prefix+'event_scores_path']=score_path
                plots=ev.plot_lepton_groups(selection,tail=tail,output_dir=root,feature_names=features,
                    feature_payload={'particle_features':features},num_events=3,seed=42)
                for group in ('retained','vetoed'):
                    self.assertTrue(plots[group]['path'].is_file())
                    self.assertTrue(plots[group]['features_path'].is_file())
                    self.assertEqual(len(plots[group]['indices']),3)
                    self.assertEqual(plots[group]['indices'],sorted(plots[group]['indices']))


if __name__=='__main__': unittest.main()
