import streamlit as st
import pandas as pd
from grouping_logic.processor import GroupProcessor


def detect_column(columns, keywords):
    """Find the most likely column from list using keyword matching"""
    columns_lower = [col.lower() for col in columns]
    for keyword in keywords:
        for i, col in enumerate(columns_lower):
            if keyword in col:
                return i
    return 0  # Fallback to first column if no matches


def auto_fill_categories(df, position_col, categories, group_size):
    """Auto-fill categories with guaranteed minimum fulfillment"""
    total_participants = len(df)
    min_members = total_participants // group_size  # Ceiling division

    # Save manual assignments
    manual_assignments = {
        cat: df[df[position_col].isin(positions)]
        for cat, positions in categories.items()
    }

    lists = {cat: df_assigned.copy() for cat, df_assigned in manual_assignments.items()}

    def move_participant(source, target):
        """Move one participant between categories"""
        if len(lists[source]) == 0:
            return False
        participant = lists[source].sample(1)
        lists[source] = lists[source].drop(participant.index)
        lists[target] = pd.concat([lists[target], participant])
        return True

    # Phase 1: Priority filling for hierarchical structure
    priority_order = [
        ('Mid-Level', 'Senior-Level'),
        ('Early Career', 'Mid-Level'),
        ('Student', 'Early Career')
    ]

    for source, target in priority_order:
        while len(lists[target]) < min_members:
            if not move_participant(source, target):
                # Fallback to any available source if preferred is empty
                potential_sources = [s for s in categories if len(lists[s]) > 0]
                if not potential_sources:
                    break
                source = max(potential_sources, key=lambda x: len(lists[x]))
                if not move_participant(source, target):
                    break

    # Phase 2: Force minimum requirements
    for cat in categories:
        while len(lists[cat]) < min_members:
            potential_sources = [s for s in categories if len(lists[s]) > min_members]
            if not potential_sources:
                potential_sources = [s for s in categories if len(lists[s]) > 0]
                if not potential_sources:
                    break
            source = max(potential_sources, key=lambda x: len(lists[x]))
            if not move_participant(source, cat):
                break

    # Final validation check
    for cat in categories:
        if len(lists[cat]) < min_members:
            st.error(f"Critical error: {cat} has {len(lists[cat])} members (min {min_members})")
            st.stop()

    return pd.concat(lists.values()), categories

def main():
    st.title("Randomized Coffee Trial Group Generator ")

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        cols = df.columns.tolist()

        # Auto-detect columns with smart defaults
        name_idx = detect_column(cols, ['name', 'full name', 'employee'])
        email_idx = detect_column(cols, ['email', 'mail'])
        position_idx = detect_column(cols, ['position', 'role', 'title'])
        job_sector_idx = detect_column(cols, ['job sector', 'sector', 'department'])

        # Column selection with auto-detected defaults
        col1, col2, col3 = st.columns(3)
        with col1:
            name_col = st.selectbox("Name Column", cols, index=name_idx)
        with col2:
            email_col = st.selectbox("Email Column", cols, index=email_idx)
        with col3:
            job_sector_col = st.selectbox("Job Sector Column", cols, index=job_sector_idx)

        position_col = st.selectbox("Position Column", cols, index=position_idx)

        # Initialize categorization state
        if 'categories' not in st.session_state:
            st.session_state.categories = {
                'Student': [],
                'Early Career': [],
                'Mid-Level': [],
                'Senior-Level': []
            }

        st.header("Categorize Positions")
        positions = df[position_col].unique().tolist()

        # Create columns for each category
        category_cols = st.columns(4)
        # In the main() function, around line 58:
        for i, category in enumerate(st.session_state.categories):
            with category_cols[i]:
                st.subheader(category)

                # Calculate available positions for this category
                used_positions = sum(st.session_state.categories.values(), [])
                available = [p for p in positions if p not in used_positions]

                # Add position to category
                add_pos = st.selectbox(
                    f"Add to {category}",
                    available,
                    key=f"add_select_{category}"
                )
                if st.button(f"Add to {category}", key=f"add_btn_{category}"):
                    st.session_state.categories[category].append(add_pos)
                    st.rerun()

                # Remove position from category
                if st.session_state.categories[category]:
                    remove_pos = st.selectbox(
                        f"Remove from {category}",
                        st.session_state.categories[category],
                        key=f"remove_select_{category}"
                    )
                    if st.button(f"Remove from {category}", key=f"remove_btn_{category}"):
                        st.session_state.categories[category].remove(remove_pos)
                        st.rerun()

                # Show current positions
                st.write("Current:", ", ".join(st.session_state.categories[category]))

        # Validation and group generation
        group_size = st.number_input(
            "Group Size",
            min_value=2,
            max_value=10,
            value=4,
            help="Number of participants per group",
            key="main_group_size"  # Unique key for single definition
        )
        min_members = group_size * 2

        valid = True
        for category, positions in st.session_state.categories.items():
            count = len(df[df[position_col].isin(positions)])
            if count < min_members:
                valid = False
                st.error(f"{category} needs at least {min_members} members (current: {count})")

        # In the main() function, update the auto-fill section:
        # In the main() function, update the auto-fill section:
        if not valid:
            if st.button("Fill Non-Filled Classes", key="auto_fill_btn"):
                df_filled, new_categories = auto_fill_categories(
                    df,
                    position_col,
                    st.session_state.categories,
                    group_size  # Pass group_size instead of min_members
                )

                # Update session state with debug info
                st.session_state.update({
                    'categories': new_categories,
                    'auto_fill_done': True,
                    'debug_data': {
                        'df_filled': df_filled.assign(
                            position_category=df_filled[position_col].apply(
                                lambda x: next(
                                    (cat for cat, positions in new_categories.items()
                                     if x in positions),
                                    'Uncategorized'
                                )
                            )
                        ),
                        'new_categories': new_categories
                    }
                })

            # Permanent debug output section
            if st.session_state.get('auto_fill_done'):
                st.success("Categories automatically filled!")

                debug_data = st.session_state.debug_data
                st.write("### Auto-Fill Results")

                for category, positions in debug_data['new_categories'].items():
                    category_df = debug_data['df_filled'][debug_data['df_filled'][position_col].isin(positions)]
                    with st.expander(f"{category} ({len(category_df)} members)"):
                        # Use unique column names for display
                        display_df = category_df[[name_col, email_col, position_col]].copy()
                        display_df.columns = ['Participant Name', 'Email Address', 'Position Category']
                        st.write(display_df)

                # In the Generate Teams button click handler:
                if st.button("Generate Randomized Teams", key="generate_teams_btn"):# When initializing GroupProcessor:
                    # Updated GroupProcessor initialization with all required parameters
                    processor = GroupProcessor(
                        df=df,
                        group_size=group_size,
                        position_col=position_col,
                        name_col=name_col,
                        email_col=email_col,
                        job_sector_col=job_sector_col
                    )
                    result = processor.generate_groups()

                    if result['success']:
                        # Create downloadable Excel file
                        output_path = "group_assignments.xlsx"
                        result['df'].to_excel(output_path, index=False)

                        st.success(f"Created {result['num_groups']} groups!")
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="Download Groups",
                                data=f,
                                file_name=output_path,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        st.dataframe(result['df'])
                    else:
                        st.error(f"Error: {result['error']}")

        if valid and st.button("Generate Groups"):
            # Map categories to DataFrame
            df['position_category'] = df[position_col].apply(
                lambda p: next(
                    (cat for cat, positions in st.session_state.categories.items() if p in positions),
                    'Uncategorized'
                )
            )

            processor = GroupProcessor(
                df=df,
                group_size=group_size,
                position_col=position_col,
                name_col=name_col,
                email_col=email_col,
                job_sector_col=job_sector_col
            )

            result = processor.generate_groups()

            # Format output with original columns
            output_data = []
            for _, row in result['df'].iterrows():
                output_data.append([
                    row[name_col],
                    row[email_col],
                    row[position_col],
                    row[job_sector_col],
                    row['Group']
                ])

            output_df = pd.DataFrame(
                output_data,
                columns=['Name', 'Email', 'Position', 'Job Sector', 'Group']
            )

            # Display results
            st.success(f"Created {len(result['df']['Group'].unique())} groups!")
            st.download_button(
                label="Download Assignments",
                data=output_df.to_csv(index=False),
                file_name="group_assignments.csv",
                mime="text/csv"
            )
            st.dataframe(output_df)


if __name__ == "__main__":
    main()