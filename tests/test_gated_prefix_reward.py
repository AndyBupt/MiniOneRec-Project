import math
import unittest
from grpo_utils import gated_prefix_signals, make_gated_prefix_reward, make_ranking_reward, rule_reward


class GatedPrefixTest(unittest.TestCase):
    target = '<a_1><b_2><c_3>'
    candidates = ['<a_1><b_2><c_4>', '<a_1><b_5><c_6>', '<a_7><b_8><c_9>', 'invalid']

    def signals(self, comps=None, task='sid_next', scale=.1):
        return gated_prefix_signals(['p']*4, comps or self.candidates,
                                    [self.target]*4, [task]*4, 4, ['s']*4, scale)

    def test_all_miss_gets_different_bonuses_and_post_zscore_scale(self):
        bonus, scales, eligible = self.signals()
        self.assertEqual(bonus, [.5, .2, 0., 0.])
        self.assertEqual(scales, [.1]*4)
        self.assertEqual(eligible, [True]*4)
        mean = sum(bonus)/4
        std = math.sqrt(sum((x-mean)**2 for x in bonus)/3)
        before = [(x-mean)/(std+1e-4) for x in bonus]
        after = [x*s for x,s in zip(before, scales)]
        self.assertAlmostEqual(sum(after), 0.)
        self.assertAlmostEqual(max(after), .1*max(before))

    def test_hit_group_and_other_tasks_preserve_original_rewards(self):
        for comps, task in [(self.candidates[:-1]+[self.target], 'sid_next'),
                            (self.candidates, 'title_history_next_sid')]:
            bonus, scales, eligible = self.signals(comps, task)
            self.assertEqual(bonus, [0.]*4)
            self.assertEqual(scales, [1.]*4)
            self.assertFalse(any(eligible))
            base = [x+y for x,y in zip(rule_reward(['p']*4, comps, [self.target]*4),
                                       make_ranking_reward(4)(['p']*4, comps, [self.target]*4))]
            self.assertEqual(base, [x+y for x,y in zip(base,bonus)])

    def test_same_prefix_still_zero_variance(self):
        bonus, _, _ = self.signals(['<a_1><b_2><c_4>']*4)
        self.assertEqual(bonus,[.5]*4)
        self.assertEqual([x-sum(bonus)/4 for x in bonus], [0.]*4)

    def test_parser_requires_complete_sid_and_correct_layer_order(self):
        bonus,_,_ = self.signals(['<a_1><b_2>', '<b_1><a_2><c_4>', '<a_10><b_2><c_4>', '<a_1><b_20><c_4>'])
        self.assertEqual(bonus,[0.,0.,0.,.2])

    def test_groups_cannot_mix_targets_or_tasks(self):
        with self.assertRaises(ValueError):
            gated_prefix_signals(['p']*4,self.candidates,[self.target]*3+['<a_9><b_2><c_3>'],['sid_next']*4,4)
        with self.assertRaises(ValueError):
            gated_prefix_signals(['p']*4,self.candidates,[self.target]*4,['sid_next']*3+['other'],4)
        with self.assertRaises(ValueError):
            self.signals(scale=float('nan'))

    def test_reward_callback_uses_sample_labels(self):
        bonus = make_gated_prefix_reward(4)(['p']*4,self.candidates,[self.target]*4,['sid_next']*4,sample_id=['s']*4)
        self.assertEqual(bonus,[.5,.2,0.,0.])


if __name__ == '__main__':
    unittest.main()
