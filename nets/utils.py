import os
from pathlib import Path
import matplotlib.pyplot as plt

def plot_video_scores(video_results, title="Frame Scores Comparison", save_path="frame_scores_comparison.png",
                      ylim=None):
    """
    Plot frame scores for multiple videos on the same graph.
    """
    plt.figure(figsize=(12, 6))
    
    # Get metric type from first result
    metric_type = next(iter(video_results.values())).get('metric_type', 'Unknown')
    
    for video_path, results in video_results.items():
        # Get the basename of the video for the legend
        video_name = os.path.basename(video_path)
        avg_score = results['avg_score']
        
        # Plot the scores
        frame_scores = results['frame_scores']
        frames = range(1, len(frame_scores) + 1)
        plt.plot(frames, frame_scores, label=f"{video_name} (avg: {avg_score:.4f})")
    
    plt.xlabel('Frame Number')
    plt.ylabel(f'Score ({metric_type})')
    plt.title(f"{title} ({metric_type})")

    if not isinstance(ylim, type(None)):
        plt.ylim(ylim)
    
    # Move the legend outside the plot to avoid covering data
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout for legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

def plot_attribute_scores(video_results, title="Attribute Scores", save_path="attribute_scores.png", ylim=None):
    """
    Plot attribute scores for multiple videos on the same graph.
    """
    # Check if attribute scores are available
    first_result = next(iter(video_results.values()))
    if 'avg_attribute_scores' not in first_result:
        print("No attribute scores available for plotting")
        return
    
    # Get all attributes
    attributes = list(first_result['avg_attribute_scores'].keys())
    
    # Create figure with subplots for each attribute
    fig, axes = plt.subplots(len(attributes), 1, figsize=(12, 4 * len(attributes)), sharex=True)
    
    # Get metric type
    metric_type = first_result.get('metric_type', 'Unknown')
    
    # Plot each attribute
    for i, attr in enumerate(attributes):
        ax = axes[i] if len(attributes) > 1 else axes
        
        for video_path, results in video_results.items():
            # Get the basename of the video for the legend
            video_name = os.path.basename(video_path)
            
            # Plot the attribute scores
            if 'attribute_scores' in results and attr in results['attribute_scores']:
                attr_scores = results['attribute_scores'][attr]
                frames = range(1, len(attr_scores) + 1)
                avg_score = results['avg_attribute_scores'][attr]
                ax.plot(frames, attr_scores, label=f"{video_name} (avg: {avg_score:.4f})")
        
        ax.set_ylabel(f"{attr.capitalize()}")
        ax.set_title(f"{attr.capitalize()} Scores")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set common labels
    fig.text(0.5, 0.04, 'Frame Number', ha='center', va='center')
    fig.text(0.06, 0.5, f'Score ({metric_type})', ha='center', va='center', rotation='vertical')
    fig.suptitle(f"{title} ({metric_type})")
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.8, top=0.95)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Attribute plot saved to {save_path}")


