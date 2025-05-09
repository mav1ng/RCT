import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import pandas as pd
import random


class GroupProcessor:
    def __init__(self, df, group_size, position_col='position_category',
                 name_col='Name', email_col='Email', job_sector_col='Job Sector'):
        """
        Initialize group processor with user-specified columns

        Args:
            df: DataFrame containing participant data
            group_size: Target number of members per group
            position_col: Column name for position categories
            name_col: User-selected name column
            email_col: User-selected email column
            job_sector_col: Column name for job sector
        """
        self.df = df
        self.group_size = group_size
        self.position_col = position_col
        self.name_col = name_col
        self.email_col = email_col
        self.job_sector_col = job_sector_col

    def generate_groups(self):
        try:
            from ortools.graph.python import linear_sum_assignment
            import numpy as np

            # Validate required columns
            required_columns = [
                self.position_col,
                self.name_col,
                self.email_col,
                self.job_sector_col
            ]

            missing = [col for col in required_columns
                       if col not in self.df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")

            # Filter out participants without category
            valid_df = self.df.dropna(subset=[self.position_col])
            participants = valid_df.to_dict('records')

            # Calculate position/sector frequencies
            position_counts = valid_df[self.position_col].value_counts().to_dict()
            sector_counts = valid_df[self.job_sector_col].value_counts().to_dict()

            # Create list of (participant, rarity_score)
            participants_with_rarity = []
            for p in participants:
                pos_rarity = 1 / position_counts[p[self.position_col]]
                sec_rarity = 1 / sector_counts[p[self.job_sector_col]]
                participants_with_rarity.append( (p, pos_rarity + sec_rarity) )
            
            # Sort by descending rarity (rarest first)
            participants_sorted = sorted(participants_with_rarity, key=lambda x: -x[1])
            sorted_indices = [i for i, (p, _) in enumerate(participants_sorted)]

            # Build cost matrix
            num_participants = len(participants)
            cost_matrix = np.zeros((num_participants, num_participants))
            
            POSITION_WEIGHTS = {
                'Senior-Level': 3,
                'Mid-Level': 2,
                'Student/Early Career': 1
            }

            for i in range(num_participants):
                for j in range(num_participants):
                    if i == j:
                        cost = 0
                    else:
                        # Position mismatch cost (higher = worse)
                        p1_pos = participants[i][self.position_col]
                        p2_pos = participants[j][self.position_col]
                        pos_cost = 100 if p1_pos == p2_pos else 0
                        
                        # Sector mismatch cost (lower priority)
                        p1_sec = participants[i][self.job_sector_col]
                        p2_sec = participants[j][self.job_sector_col]
                        sec_cost = 33 if p1_sec == p2_sec else 0
                        
                        cost = pos_cost + sec_cost
                    cost_matrix[i][j] = cost

            # Solve assignment problem
            assignment = linear_sum_assignment.SimpleLinearSumAssignment()
            for worker in range(num_participants):
                for task in range(num_participants):
                    if cost_matrix[worker][task] < 1000:
                        assignment.add_arc_with_cost(worker, task, int(cost_matrix[worker][task]))
            solve_status = assignment.solve()

            if solve_status != assignment.OPTIMAL:
                raise RuntimeError("Optimal assignment not found")

            # Form groups from assignment
            groups = []
            group_size = self.group_size
            
            # Get optimal assignment order
            sorted_indices = []
            for i in range(assignment.num_nodes()):
                sorted_indices.append(assignment.right_mate(i))
            
            # Create groups in optimized order
            for i in range(0, len(sorted_indices), group_size):
                group_indices = sorted_indices[i:i+group_size]
                group_members = [participants[idx] for idx in group_indices]
                
                groups.append({
                    'Group ID': f"Group-{len(groups)+1}",
                    'Members': group_members
                })

            # Calculate remainder
            remainder = len(participants) % group_size

            # Assign remainders to maximize diversity
            if remainder > 0:
                # Get rarest members
                rare_members = [participants[idx] for idx in sorted_indices[:remainder]]
                
                # Assign each rare member to the group where they add most diversity
                for rare in rare_members:
                    best_group = None
                    best_score = -1
                    for group in groups:
                        current_pos = set(m[self.position_col] for m in group['Members'])
                        current_sec = set(m[self.job_sector_col] for m in group['Members'])
                        
                        new_pos = current_pos.union({rare[self.position_col]})
                        new_sec = current_sec.union({rare[self.job_sector_col]})
                        
                        score = (len(new_pos) - len(current_pos)) * 100 + \
                                (len(new_sec) - len(current_sec)) * 33
                        if score > best_score:
                            best_score = score
                            best_group = group
                    
                    best_group['Members'].append(rare)

            # Build output DataFrame
            output_data = []
            for group in groups:
                for member in group['Members']:
                    output_data.append({
                        'Group': group['Group ID'],
                        self.name_col: member[self.name_col],
                        self.email_col: member[self.email_col],
                        self.position_col: member[self.position_col],
                        self.job_sector_col: member[self.job_sector_col]
                    })

            output_df = pd.DataFrame(output_data)

            return {
                'success': True,
                'num_groups': len(groups),
                'df': output_df
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _validate_position_categories(self):
        valid_categories = {'Student', 'Early Career', 'Mid-Level', 'Senior-Level'}
        invalid = set(self.df[self.position_col]) - valid_categories
        if invalid:
            raise ValueError(f"Invalid position categories detected: {invalid}")


    def merge_classes(self, old_class_1, old_class_2, new_class_name):
        """Merge two classes into a new class."""
        # Assuming 'position_merged' is the column where classes are defined
        self.df.loc[self.df['position_merged'] == old_class_1, 'position_merged'] = new_class_name
        self.df.loc[self.df['position_merged'] == old_class_2, 'position_merged'] = new_class_name
        # You may want to do the same for job_family_merged or other relevant columns

    def identify_small_classes(self):
        """Identify classes that can't support the required number of groups"""
        num_groups = max(len(self.df) // self.group_size, 1)
        min_per_class = num_groups  # Each class needs at least 1 member per group
        
        position_counts = self.df['position_merged'].value_counts()
        small_positions = position_counts[position_counts < min_per_class].index.tolist()
        
        job_family_counts = self.df['job_family_merged'].value_counts()
        small_job_families = job_family_counts[job_family_counts < min_per_class].index.tolist()
        
        return small_positions, small_job_families

    def _validate_columns(self, *cols):
        missing = [col for col in cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")

    def _rename_columns(self, position_col, job_family_col, name_col, email_col):
        rename_dict = {}
        if position_col != "Position":
            rename_dict[position_col] = "Position"
        if job_family_col != "Job Family":
            rename_dict[job_family_col] = "Job Family"
        if name_col != "Name":
            rename_dict[name_col] = "Name"
        if email_col != "Email":
            rename_dict[email_col] = "Email"
        self.df = self.df.rename(columns=rename_dict)

    def _preprocess_data(self):
        self.df['position_merged'] = self.df['Position'].apply(self._map_position)
        self.df['job_family_merged'] = self.df['Job Family'].apply(self._map_job_family)
        # Ensure that the merged columns are categorized properly for stratification
        self.df['stratify_group'] = self.df['position_merged'] + "_" + self.df['job_family_merged']

    def _map_position(self, position):
        position = str(position).lower()
        if 'student' in position or 'early career' in position:
            return 'Student/Early Career'
        elif 'senior' in position or '>10' in position:
            return 'Senior-level'
        elif 'mid' in position or '3-10' in position:
            return 'Mid-level'
        return 'Entry-level'

    def _map_job_family(self, job_family):
        job_family = str(job_family).lower()
        return job_family.title()

def identify_small_classes(self, threshold=5):
    """Identify classes in position and job family that are below the threshold."""
    position_counts = self.df['position_merged'].value_counts()
    job_family_counts = self.df['job_family_merged'].value_counts()

    small_positions = position_counts[position_counts < threshold].index.tolist()
    small_job_families = job_family_counts[job_family_counts < threshold].index.tolist()

    return small_positions, small_job_families

def merge_classes(self, old_class_1, old_class_2, new_class_name):
    """Merge two classes into a new class."""
    self.df['position_merged'] = self.df['position_merged'].replace({old_class_1: new_class_name, old_class_2: new_class_name})
    self.df['job_family_merged'] = self.df['job_family_merged'].replace({old_class_1: new_class_name, old_class_2: new_class_name})