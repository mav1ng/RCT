import pandas as pd
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
import copy


class Processor:
    def __init__(self, df, group_size=5, num_groups=None, position_col='Position_Category',
                 name_col='Name', email_col='Email', job_sector_col='Job_Sector'):
        """
        Initialize group processor with user-specified columns

        Args:
            df: DataFrame containing participant data
            group_size: Target number of members per group
            num_groups: Number of groups to form (optional)
            position_col: Column name for position categories
            name_col: User-selected name column
            email_col: User-selected email column
            job_sector_col: Column name for job sector
        """
        if position_col not in df.columns:
            raise ValueError(f"Position column '{position_col}' not found in DataFrame")
        
        required_columns = [
            position_col,
            name_col,
            email_col,
            job_sector_col
        ]

        missing = [col for col in required_columns
                   if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.df = df
        self.group_size = group_size
        self.num_groups = num_groups or len(df) // group_size
        self.position_col = position_col
        self.name_col = name_col
        self.email_col = email_col
        self.job_sector_col = job_sector_col

    def generate_groups(self):
        """
        Generate groups based on stratified split (if possible), then greedy grouping for leftovers.
        Ensures all groups have min_size = group_size and max_size = group_size + 1.
        Returns a dict with groups, DataFrame, total diversity score, and best swap suggestion (if any).
        """
        try:
            required_columns = [
                self.position_col,
                self.name_col,
                self.email_col,
                self.job_sector_col
            ]
            missing = [col for col in required_columns if col not in self.df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            self.df['index'] = self.df.index

            # Step 1: Robust stratified split
            stratified_groups, stratified_indices = self._robust_stratified_split()

            # Step 2: Greedy grouping for leftovers
            leftovers = self.df[~self.df['index'].isin(stratified_indices)]
            leftover_groups = self._greedy_grouping(leftovers)

            all_groups = stratified_groups + leftover_groups

            # Step 3: Final validation
            for g in all_groups:
                if not (self.group_size <= len(g['Members']) <= self.group_size + 1):
                    # Allow a single undersized leftover group
                    if not (g['Group ID'].startswith('Leftover-') and len(g['Members']) < self.group_size):
                        raise ValueError(f"Group {g['Group ID']} does not meet size requirements: {len(g['Members'])}")

            # Track assignment method for each member
            stratified_ids = {m['index'] for g in stratified_groups for m in g['Members']}
            greedy_ids = {m['index'] for g in leftover_groups for m in g['Members']}
            leftover_ids = set()
            if leftover_groups:
                last_group = leftover_groups[-1]
                if last_group['Group ID'].startswith('Leftover-') and len(last_group['Members']) < self.group_size:
                    leftover_ids = {m['index'] for m in last_group['Members']}
                    greedy_ids -= leftover_ids

            result_df = self._format_results(
                all_groups,
                stratified_ids=stratified_ids,
                greedy_ids=greedy_ids,
                leftover_ids=leftover_ids
            )
            total_div_score = self.total_diversity_score(all_groups)
            swap = self.find_best_swap(all_groups)

            return {
                'success': True,
                'message': 'Groups generated successfully',
                'df': result_df,
                'groups': all_groups,
                'num_groups': len(all_groups),
                'diversity_score': total_div_score,
                'swap_suggestion': swap
            }
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return {
                'success': False,
                'message': f'Error generating groups: {str(e)}',
                'df': pd.DataFrame(),
                'groups': []
            }
        finally:
            if 'index' in self.df.columns:
                self.df.drop('index', axis=1, inplace=True)

    def _greedy_grouping(self, df):
        """
        Greedy grouping for leftovers. Only create fully filled groups (group_size),
        then distribute any leftover participants to existing groups (so all groups have size in [group_size, group_size+1]).
        If leftovers remain after distributing, create one final group with them (even if < group_size).
        Ensures all participants are assigned.
        """
        if df.empty:
            return []
        df = df.copy()
        df['index'] = df.index
        n = len(df)
        groups = []
        indices = list(df['index'])
        i = 0
        group_num = 1
        # Step 1: Create full groups
        while i + self.group_size <= n:
            group_indices = indices[i:i + self.group_size]
            members = [df[df['index'] == idx].iloc[0].to_dict() for idx in group_indices]
            groups.append({'Group ID': f"Greedy-{group_num}", 'Members': members})
            i += self.group_size
            group_num += 1
        # Step 2: Distribute leftovers to existing groups
        leftovers = indices[i:]
        if groups and leftovers:
            for j, idx in enumerate(leftovers):
                groups[j % len(groups)]['Members'].append(df[df['index'] == idx].iloc[0].to_dict())
        elif leftovers:
            # Not enough for even one full group: create a single leftover group
            members = [df[df['index'] == idx].iloc[0].to_dict() for idx in leftovers]
            groups.append({'Group ID': f"Leftover-1", 'Members': members})
        return groups

    def _robust_stratified_split(self):
        """
        For each group, pick one member from each class (as many as there are classes),
        then fill the remaining slots with members from any class, balancing class distribution.
        Always create groups of size group_size, as long as enough participants are available.
        Returns (groups, assigned_indices)
        """
        if 'index' not in self.df.columns:
            self.df['index'] = self.df.index
        categories = self.df[self.position_col].unique().tolist()
        n_cats = len(categories)
        print(f"[DEBUG] Stratified split using position_col='{self.position_col}':")
        print(self.df[self.position_col].value_counts().to_dict())
        # Find how many full groups we can make
        cat_counts = [len(self.df[self.df[self.position_col] == cat]) for cat in categories]
        # Number of groups is limited by total participants and class counts
        max_groups = min(len(self.df) // self.group_size, min(cat_counts))
        if max_groups == 0:
            print("[DEBUG] No groups possible for stratified split (empty class or not enough participants)")
            return [], set()
        groups = [{'Group ID': f"Group-{i+1}", 'Members': []} for i in range(max_groups)]
        assigned_indices = set()
        # Prepare a pool for each class
        class_pools = {cat: self.df[(self.df[self.position_col] == cat) & (~self.df['index'].isin(assigned_indices))].copy() for cat in categories}
        # Prepare a pool for all unassigned
        all_pool = self.df[~self.df['index'].isin(assigned_indices)].copy()
        for i in range(max_groups):
            group_members = []
            used_this_group = set()
            # Step 1: Pick one from each class (as many as there are classes)
            for cat in categories:
                pool = class_pools[cat][~class_pools[cat]['index'].isin(assigned_indices | used_this_group)]
                if not pool.empty:
                    selected = pool.sample(1, random_state=i).iloc[0].to_dict()
                    group_members.append(selected)
                    assigned_indices.add(selected['index'])
                    used_this_group.add(selected['index'])
            # Step 2: Fill remaining slots
            while len(group_members) < self.group_size:
                # Use all remaining unassigned participants
                all_pool = self.df[~self.df['index'].isin(assigned_indices | used_this_group)]
                if all_pool.empty:
                    break
                # Prefer classes with most remaining members
                largest_class = all_pool[self.position_col].value_counts().idxmax()
                pool = all_pool[all_pool[self.position_col] == largest_class]
                selected = pool.sample(1, random_state=i*10+len(group_members)).iloc[0].to_dict()
                group_members.append(selected)
                assigned_indices.add(selected['index'])
                used_this_group.add(selected['index'])
            if len(group_members) == self.group_size:
                groups[i]['Members'] = group_members
            else:
                groups[i]['Members'] = []  # Incomplete group, ignore
        final_groups = [g for g in groups if len(g['Members']) == self.group_size]
        final_indices = set(idx for g in final_groups for idx in [m['index'] for m in g['Members']])
        print(f"[DEBUG] Stratified split assigned {len(final_indices)} participants to {len(final_groups)} groups.")
        return final_groups, final_indices

    def _diversity_score(self, group_members, candidate):
        """
        Score for adding candidate to group_members (more unique positions/sectors is better)
        """
        positions = set(m[self.position_col] for m in group_members)
        sectors = set(m[self.job_sector_col] for m in group_members)
        score = 0
        if candidate[self.position_col] not in positions:
            score += 100
        if candidate[self.job_sector_col] not in sectors:
            score += 50
        return score

    def group_diversity_score(self, group):
        """
        Diversity score for a group: +100 per unique position, -50 per duplicate position.
        """
        members = group['Members']
        positions = [m[self.position_col] for m in members]
        unique_positions = len(set(positions))
        duplicates = len(positions) - unique_positions
        return unique_positions * 100 - duplicates * 50

    def total_diversity_score(self, groups):
        return sum(self.group_diversity_score(g) for g in groups)

    def find_best_swap(self, groups):
        """
        Try all possible single swaps between groups, return the swap that increases total diversity score the most (if any).
        Returns a dict: {'score_gain': int, 'swap': (gid1, idx1, name1, gid2, idx2, name2)}, or None if no improving swap.
        """
        best_gain = 0
        best_swap = None
        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups):
                if i >= j:
                    continue
                for m1 in g1['Members']:
                    for m2 in g2['Members']:
                        # Swap m1 and m2
                        g1_members = [m for m in g1['Members'] if m['index'] != m1['index']] + [m2]
                        g2_members = [m for m in g2['Members'] if m['index'] != m2['index']] + [m1]
                        # Check size constraints
                        if not (self.group_size <= len(g1_members) <= self.group_size + 1):
                            continue
                        if not (self.group_size <= len(g2_members) <= self.group_size + 1):
                            continue
                        g1_score = self.group_diversity_score({'Members': g1_members})
                        g2_score = self.group_diversity_score({'Members': g2_members})
                        old_score = self.group_diversity_score(g1) + self.group_diversity_score(g2)
                        gain = (g1_score + g2_score) - old_score
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = {
                                'score_gain': gain,
                                'swap': (g1.get('Group ID'), m1['index'], m1.get(self.name_col),
                                         g2.get('Group ID'), m2['index'], m2.get(self.name_col))
                            }
        return best_swap

    def _format_results(self, groups, stratified_ids=None, greedy_ids=None, leftover_ids=None):
        """
        Format the groups into a DataFrame for display and export.
        Adds an Assignment_Method column: 'Stratified Split', 'Greedy Assignment', or 'Leftover Greedy Assignment'.
        The 'Group' column is a simple integer (1, 2, ...) for each group.
        """
        all_members = []
        for group_num, group in enumerate(groups, start=1):
            for member in group['Members']:
                member_dict = {k: v for k, v in member.items() if k != 'index'}
                member_dict['Group'] = group_num
                idx = member['index']
                if stratified_ids and idx in stratified_ids:
                    member_dict['Assignment_Method'] = 'Stratified Split'
                elif leftover_ids and idx in leftover_ids:
                    member_dict['Assignment_Method'] = 'Leftover Greedy Assignment'
                elif greedy_ids and idx in greedy_ids:
                    member_dict['Assignment_Method'] = 'Greedy Assignment'
                else:
                    member_dict['Assignment_Method'] = 'Unknown'
                all_members.append(member_dict)
        result_df = pd.DataFrame(all_members)
        return result_df

    def assign_groups(self):
        """Legacy method for backward compatibility"""
        return self.generate_groups()['df']

    def optimize_swaps_for_diversity(self, groups):
        """
        Repeatedly apply the best available swap until no further improvement in diversity score is possible.
        Returns (optimized_groups, score_before, score_after, swaps_performed)
        """
        import copy
        groups = copy.deepcopy(groups)
        score_before = self.total_diversity_score(groups)
        swaps_performed = []
        while True:
            swap = self.find_best_swap(groups)
            if not swap or swap['score_gain'] <= 0:
                break
            # Apply the swap
            gid1, idx1, _, gid2, idx2, _ = swap['swap']
            g1 = next(g for g in groups if g['Group ID'] == gid1)
            g2 = next(g for g in groups if g['Group ID'] == gid2)
            m1 = next(m for m in g1['Members'] if m['index'] == idx1)
            m2 = next(m for m in g2['Members'] if m['index'] == idx2)
            # Swap them
            g1['Members'] = [m2 if m['index'] == idx1 else m for m in g1['Members']]
            g2['Members'] = [m1 if m['index'] == idx2 else m for m in g2['Members']]
            swaps_performed.append(swap)
        score_after = self.total_diversity_score(groups)
        return groups, score_before, score_after, swaps_performed