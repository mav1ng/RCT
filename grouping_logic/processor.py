import pandas as pd
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
import copy


class Processor:
    def __init__(self, df, group_size=5, num_groups=None, position_col='Position_Category',
                 name_col='Name', email_col='Email', job_sector_col='Job_Sector', seed=42):
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
            seed: Random seed for reproducibility
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
        self.seed = seed

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
            # New: If there is a leftover group with < group_size, reassign those members to groups maximizing diversity
            if leftover_groups:
                last_group = leftover_groups[-1]
                if last_group['Group ID'].startswith('Leftover-') and len(last_group['Members']) < self.group_size:
                    leftover_members = last_group['Members']
                    # Remove the leftover group from all_groups and leftover_groups
                    all_groups = stratified_groups + leftover_groups[:-1]
                    leftover_groups = leftover_groups[:-1]
                    # Assign each leftover member to the group (not exceeding group_size+1) maximizing diversity
                    for member in leftover_members:
                        best_score = float('-inf')
                        best_group = None
                        for group in all_groups:
                            if len(group['Members']) < self.group_size + 1:
                                score = self._diversity_score(group['Members'], member)
                                if score > best_score:
                                    best_score = score
                                    best_group = group
                        if best_group is not None:
                            best_group['Members'].append(member)
                            greedy_ids.add(member['index'])
                        else:
                            # If all groups are full, just append to the last group
                            all_groups[-1]['Members'].append(member)
                            greedy_ids.add(member['index'])
                    leftover_ids = set()  # All leftovers now assigned
                    # No leftover group remains
            # Continue as before
            result_df = self._format_results(all_groups, stratified_ids=stratified_ids, greedy_ids=greedy_ids, leftover_ids=leftover_ids)
            total_score = self.total_diversity_score(all_groups)
            swap_suggestion = self.find_best_swap(all_groups)
            return {
                'success': True,
                'groups': all_groups,
                'df': result_df,
                'diversity_score': total_score,
                'swap_suggestion': swap_suggestion,
                'num_groups': len(all_groups),
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
        Greedy grouping for leftovers. Assigns each participant to the group where they maximize the diversity score.
        Only create fully filled groups (group_size), then distribute any leftover participants to existing groups (so all groups have size in [group_size, group_size+1]).
        If leftovers remain after distributing, create one final group with them (even if < group_size).
        Ensures all participants are assigned.
        """
        if df.empty:
            return []
        df = df.copy()
        df['index'] = df.index
        indices = list(df['index'])
        random.Random(self.seed).shuffle(indices)  # Shuffle for fairness
        groups = []
        group_num = 1
        # Step 1: Create empty groups up to needed size
        total_needed = len(indices)
        num_groups = total_needed // self.group_size
        for _ in range(num_groups):
            groups.append({'Group ID': f"Greedy-{group_num}", 'Members': []})
            group_num += 1
        # Step 2: Greedy assignment for diversity
        for idx in indices:
            participant = df[df['index'] == idx].iloc[0].to_dict()
            # Find the best group to add this participant to (not exceeding group_size+1)
            best_score = float('-inf')
            best_group = None
            for group in groups:
                if len(group['Members']) < self.group_size:
                    score = self._diversity_score(group['Members'], participant)
                    if score > best_score:
                        best_score = score
                        best_group = group
            if best_group is not None:
                best_group['Members'].append(participant)
            else:
                # All groups are full, create a leftover group
                leftover_group = next((g for g in groups if g['Group ID'].startswith('Leftover-')), None)
                if leftover_group is None:
                    leftover_group = {'Group ID': f"Leftover-1", 'Members': []}
                    groups.append(leftover_group)
                leftover_group['Members'].append(participant)
        return groups

    def _robust_stratified_split(self):
        """
        For each group, pick one member from each class (as many as there are classes),
        then fill the remaining slots with members from any class, balancing class distribution.
        Always create groups of size group_size, as long as enough participants are available.
        Returns (groups, assigned_indices)
        """
        import numpy as np
        if 'index' not in self.df.columns:
            self.df['index'] = self.df.index
        categories = self.df[self.position_col].unique().tolist()
        cat_counts = [len(self.df[self.df[self.position_col] == cat]) for cat in categories]
        min_cat = min(cat_counts)
        max_total_groups = len(self.df) // self.group_size
        total_needed = min_cat * self.group_size
        # Check if stratified split is possible
        if min_cat == 0 or max_total_groups == 0 or total_needed > len(self.df):
            print("[DEBUG] No groups possible for stratified split (empty class, not enough participants, or not enough to fill groups)")
            return [], set()
        # Each class must have at least min_cat members
        if any(count < min_cat for count in cat_counts):
            print("[DEBUG] Not enough members in at least one class for stratified split")
            return [], set()
        # 1. Select stratified subset: min_cat * group_size participants, at least min_cat from each class
        stratified_indices = []
        for cat in categories:
            cat_indices = self.df[self.df[self.position_col] == cat]['index'].sample(min_cat, random_state=self.seed).tolist()
            stratified_indices.extend(cat_indices)
        if len(stratified_indices) < min_cat * self.group_size:
            cat_indices = self.df[~self.df['index'].isin(stratified_indices)]['index'].sample(min_cat * self.group_size - len(stratified_indices),
                                                                                     random_state=self.seed).tolist()
            stratified_indices.extend(cat_indices)
        stratified_df = self.df[self.df['index'].isin(stratified_indices)].copy()
        stratified_df = stratified_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        print(stratified_df)
        # If not enough for full groups, fallback
        if len(stratified_df) < total_needed:
            print("[DEBUG] Not enough stratified participants for full groups")
            return [], set()
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=min_cat, shuffle=True, random_state=self.seed)
        groups = []
        assigned_indices = set()
        X = np.zeros((len(stratified_df), 1))
        y = stratified_df[self.position_col].values
        for group_id, (_, test_idx) in enumerate(skf.split(X, y)):
            members = stratified_df.iloc[test_idx]
            if len(members) < self.group_size:
                continue  # Only full groups
            group = {
                'Group ID': f"Group-{group_id+1}",
                'Members': members.head(self.group_size).to_dict('records')
            }
            groups.append(group)
            assigned_indices.update(members.head(self.group_size)['index'].tolist())
        # Only return if enough full groups were formed
        if len(groups) < min_cat:
            print("[DEBUG] Could not form enough full stratified groups")
            return [], set()
        return groups, assigned_indices

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