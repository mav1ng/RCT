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
        """
        Generate randomized groups with job sector information

        Returns:
            dict: {
                'success': bool,
                'num_groups': int,
                'df': DataFrame with groups,
                'error': str (if unsuccessful)
            }
        """
        try:
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

            # Create randomized groups
            groups = []
            participants = valid_df.sample(frac=1).reset_index(drop=True)

            for i in range(0, len(participants), self.group_size):
                group = participants.iloc[i:i + self.group_size]
                groups.append({
                    'Group ID': f"Group-{len(groups) + 1}",
                    'Members': group[[self.name_col, self.email_col, self.position_col, self.job_sector_col]].to_dict('records')
                })

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