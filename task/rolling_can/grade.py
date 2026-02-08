import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

def load_and_clean_data(human_file, solution_file):
    """
    Load and clean the CSV data from both files
    """
    # Load the data
    human_df = pd.read_csv(human_file)
    solution_df = pd.read_csv(solution_file)
    
    # Clean the data - remove rows with empty values
    human_df = human_df.dropna()
    solution_df = solution_df.dropna()
    
    # Convert to numeric
    for col in ['X', 'Y', 'Radius']:
        human_df[col] = pd.to_numeric(human_df[col], errors='coerce')
        solution_df[col] = pd.to_numeric(solution_df[col], errors='coerce')
    
    # Remove any remaining NaN values
    human_df = human_df.dropna()
    solution_df = solution_df.dropna()
    
    return human_df, solution_df

def calculate_distance(x1, y1, x2, y2):
    """
    Calculate Euclidean distance between two points
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def evaluate_solution(human_df, solution_df):
    """
    Evaluate how close the solution is to the human results
    """
    results = {}
    
    # Find common frames between human and solution data
    common_frames = set(human_df['Frame_Index']) & set(solution_df['Frame_Index'])
    common_frames = sorted(list(common_frames))
    
    if not common_frames:
        print("No common frames found between human and solution data!")
        return results
    
    print(f"Found {len(common_frames)} common frames for comparison")
    
    # Initialize lists to store metrics
    distances = []
    radius_diffs = []
    x_diffs = []
    y_diffs = []
    
    matched_frames = 0
    
    for frame in common_frames:
        human_frame = human_df[human_df['Frame_Index'] == frame]
        solution_frame = solution_df[solution_df['Frame_Index'] == frame]
        
        # For each human detection, find the closest solution detection
        for _, human_row in human_frame.iterrows():
            min_distance = float('inf')
            best_solution = None
            
            for _, solution_row in solution_frame.iterrows():
                # Calculate distance between circle centers
                dist = calculate_distance(
                    human_row['X'], human_row['Y'],
                    solution_row['X'], solution_row['Y']
                )
                
                if dist < min_distance:
                    min_distance = dist
                    best_solution = solution_row
            
            if best_solution is not None:
                matched_frames += 1
                distances.append(min_distance)
                radius_diffs.append(abs(human_row['Radius'] - best_solution['Radius']))
                x_diffs.append(abs(human_row['X'] - best_solution['X']))
                y_diffs.append(abs(human_row['Y'] - best_solution['Y']))
    
    # Calculate statistics
    if distances:
        results['total_frames_analyzed'] = len(common_frames)
        results['matched_detections'] = matched_frames
        results['match_rate'] = matched_frames / len(common_frames) * 100
        
        # Distance metrics
        results['mean_center_distance'] = np.mean(distances)
        results['median_center_distance'] = np.median(distances)
        results['std_center_distance'] = np.std(distances)
        results['max_center_distance'] = np.max(distances)
        results['min_center_distance'] = np.min(distances)
        
        # Radius difference metrics
        results['mean_radius_diff'] = np.mean(radius_diffs)
        results['median_radius_diff'] = np.median(radius_diffs)
        results['std_radius_diff'] = np.std(radius_diffs)
        results['max_radius_diff'] = np.max(radius_diffs)
        
        # Coordinate difference metrics
        results['mean_x_diff'] = np.mean(x_diffs)
        results['mean_y_diff'] = np.mean(y_diffs)
        results['std_x_diff'] = np.std(x_diffs)
        results['std_y_diff'] = np.std(y_diffs)
        
        # Overall accuracy score (lower is better)
        # Weighted combination of distance and radius accuracy
        distance_score = results['mean_center_distance'] / 100  # Normalize to 0-1
        radius_score = results['mean_radius_diff'] / 50  # Normalize to 0-1
        results['overall_accuracy_score'] = 1 - (distance_score + radius_score) / 2
        
    return results

def print_results(results):
    """
    Print the evaluation results in a formatted way
    """
    print("\n" + "="*60)
    print("SOLUTION EVALUATION RESULTS")
    print("="*60)
    
    if not results:
        print("No results to display!")
        return
    
    print(f"\n📊 FRAME ANALYSIS:")
    print(f"   Total frames analyzed: {results.get('total_frames_analyzed', 0)}")
    print(f"   Matched detections: {results.get('matched_detections', 0)}")
    print(f"   Match rate: {results.get('match_rate', 0):.2f}%")
    
    print(f"\n📍 CENTER DETECTION ACCURACY:")
    print(f"   Mean center distance: {results.get('mean_center_distance', 0):.2f} pixels")
    print(f"   Median center distance: {results.get('median_center_distance', 0):.2f} pixels")
    print(f"   Standard deviation: {results.get('std_center_distance', 0):.2f} pixels")
    print(f"   Min distance: {results.get('min_center_distance', 0):.2f} pixels")
    print(f"   Max distance: {results.get('max_center_distance', 0):.2f} pixels")
    
    print(f"\n🔴 RADIUS DETECTION ACCURACY:")
    print(f"   Mean radius difference: {results.get('mean_radius_diff', 0):.2f} pixels")
    print(f"   Median radius difference: {results.get('median_radius_diff', 0):.2f} pixels")
    print(f"   Standard deviation: {results.get('std_radius_diff', 0):.2f} pixels")
    print(f"   Max radius difference: {results.get('max_radius_diff', 0):.2f} pixels")
    
    print(f"\n📐 COORDINATE ACCURACY:")
    print(f"   Mean X difference: {results.get('mean_x_diff', 0):.2f} pixels")
    print(f"   Mean Y difference: {results.get('mean_y_diff', 0):.2f} pixels")
    print(f"   X standard deviation: {results.get('std_x_diff', 0):.2f} pixels")
    print(f"   Y standard deviation: {results.get('std_y_diff', 0):.2f} pixels")
    
    print(f"\n🎯 OVERALL ACCURACY SCORE:")
    accuracy = results.get('overall_accuracy_score', 0)
    print(f"   Score: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Grade the solution
    if accuracy >= 0.9:
        grade = "A+"
    elif accuracy >= 0.8:
        grade = "A"
    elif accuracy >= 0.7:
        grade = "B+"
    elif accuracy >= 0.6:
        grade = "B"
    elif accuracy >= 0.5:
        grade = "C+"
    elif accuracy >= 0.4:
        grade = "C"
    elif accuracy >= 0.3:
        grade = "D"
    else:
        grade = "F"
    
    print(f"   Grade: {grade}")
    
    print("\n" + "="*60)

def create_comparison_plots(human_df, solution_df, results):
    """
    Create visualization plots to compare the results
    """
    try:
        # Create a figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Frame coverage comparison
        human_frames = set(human_df['Frame_Index'])
        solution_frames = set(solution_df['Frame_Index'])
        all_frames = sorted(list(human_frames | solution_frames))
        
        human_coverage = [1 if frame in human_frames else 0 for frame in all_frames]
        solution_coverage = [1 if frame in solution_frames else 0 for frame in all_frames]
        
        ax1.plot(all_frames, human_coverage, 'g-', label='Human Detection', linewidth=2)
        ax1.plot(all_frames, solution_coverage, 'b-', label='Solution Detection', linewidth=2)
        ax1.set_title('Frame Detection Coverage')
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Detection (1=Detected, 0=Not Detected)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Radius comparison for common frames
        common_frames = sorted(list(human_frames & solution_frames))
        if common_frames:
            human_radii = []
            solution_radii = []
            frame_indices = []
            
            for frame in common_frames[:50]:  # Limit to first 50 for clarity
                human_frame = human_df[human_df['Frame_Index'] == frame]
                solution_frame = solution_df[solution_df['Frame_Index'] == frame]
                
                if not human_frame.empty and not solution_frame.empty:
                    human_radii.append(human_frame.iloc[0]['Radius'])
                    solution_radii.append(solution_frame.iloc[0]['Radius'])
                    frame_indices.append(frame)
            
            if frame_indices:
                ax2.plot(frame_indices, human_radii, 'g-o', label='Human Radius', markersize=4)
                ax2.plot(frame_indices, solution_radii, 'b-s', label='Solution Radius', markersize=4)
                ax2.set_title('Radius Comparison (First 50 Common Frames)')
                ax2.set_xlabel('Frame Index')
                ax2.set_ylabel('Radius (pixels)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # Plot 3: Center position comparison
        if common_frames:
            human_x = []
            human_y = []
            solution_x = []
            solution_y = []
            frame_indices = []
            
            for frame in common_frames[:50]:  # Limit to first 50 for clarity
                human_frame = human_df[human_df['Frame_Index'] == frame]
                solution_frame = solution_df[solution_df['Frame_Index'] == frame]
                
                if not human_frame.empty and not solution_frame.empty:
                    human_x.append(human_frame.iloc[0]['X'])
                    human_y.append(human_frame.iloc[0]['Y'])
                    solution_x.append(solution_frame.iloc[0]['X'])
                    solution_y.append(solution_frame.iloc[0]['Y'])
                    frame_indices.append(frame)
            
            if frame_indices:
                ax3.plot(frame_indices, human_x, 'g-o', label='Human X', markersize=4)
                ax3.plot(frame_indices, solution_x, 'b-s', label='Solution X', markersize=4)
                ax3.set_title('X-Coordinate Comparison (First 50 Common Frames)')
                ax3.set_xlabel('Frame Index')
                ax3.set_ylabel('X-Coordinate (pixels)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # Plot 4: Accuracy metrics summary
        if results:
            metrics = ['Mean Center\nDistance', 'Mean Radius\nDifference', 'Match Rate (%)']
            values = [
                results.get('mean_center_distance', 0),
                results.get('mean_radius_diff', 0),
                results.get('match_rate', 0)
            ]
            
            bars = ax4.bar(metrics, values, color=['red', 'orange', 'green'])
            ax4.set_title('Key Accuracy Metrics')
            ax4.set_ylabel('Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                        f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('solution_evaluation_plots.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved as 'solution_evaluation_plots.png'")
        
    except Exception as e:
        print(f"Could not create plots: {e}")

def main():
    """
    Main function to run the evaluation
    """
    print("Starting solution evaluation...")
    
    # Load and clean data
    try:
        human_df, solution_df = load_and_clean_data('human_circle_coordinates.csv', 'solution_circle_coordinates.csv')
        print(f"✅ Loaded {len(human_df)} human detections and {len(solution_df)} solution detections")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Evaluate the solution
    results = evaluate_solution(human_df, solution_df)
    
    # Print results
    print_results(results)
    
    # Create visualization
    # create_comparison_plots(human_df, solution_df, results)
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()
