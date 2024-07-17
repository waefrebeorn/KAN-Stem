import tensorflow as tf

def parse_event_file(event_file, output_file, max_entries_per_tag=10):
    tag_data = {}
    for event in tf.compat.v1.train.summary_iterator(event_file):
        for value in event.summary.value:
            if value.tag not in tag_data:
                tag_data[value.tag] = []
            if len(tag_data[value.tag]) < max_entries_per_tag:
                tag_data[value.tag].append((event.step, value.simple_value))

    with open(output_file, 'w') as f:
        current_epoch = None
        epoch_metrics = {"sir": [], "sdr": [], "sar": [], "g_loss": [], "d_loss": [], "val_loss": []}
        segment_metrics = {"sir": [], "sdr": [], "sar": [], "g_loss": [], "d_loss": [], "val_loss": []}
        
        for tag, entries in tag_data.items():
            for step, value in entries:
                epoch = step // 120  # Assuming 120 segments per epoch
                if current_epoch is None:
                    current_epoch = epoch
                if epoch != current_epoch:
                    # Write summary for the completed epoch
                    write_epoch_summary(f, current_epoch, epoch_metrics, segment_metrics)
                    # Reset for the new epoch
                    current_epoch = epoch
                    epoch_metrics = {"sir": [], "sdr": [], "sar": [], "g_loss": [], "d_loss": [], "val_loss": []}
                    segment_metrics = {"sir": [], "sdr": [], "sar": [], "g_loss": [], "d_loss": [], "val_loss": []}
                # Collect segment metrics
                collect_segment_metrics(segment_metrics, step, tag, value)
                # Update epoch metrics
                if "sir" in tag.lower():
                    epoch_metrics["sir"].append(value)
                elif "sdr" in tag.lower():
                    epoch_metrics["sdr"].append(value)
                elif "sar" in tag.lower():
                    epoch_metrics["sar"].append(value)
                elif "g_loss" in tag.lower():
                    epoch_metrics["g_loss"].append(value)
                elif "d_loss" in tag.lower():
                    epoch_metrics["d_loss"].append(value)
                elif "val_loss" in tag.lower():
                    epoch_metrics["val_loss"].append(value)
        # Write summary for the last epoch
        write_epoch_summary(f, current_epoch, epoch_metrics, segment_metrics)

def collect_segment_metrics(metrics, step, tag, value):
    segment = step % 120 + 1
    if "sir" in tag.lower():
        metrics["sir"].append(f"{segment},{value:.4f}")
    elif "sdr" in tag.lower():
        metrics["sdr"].append(f"{segment},{value:.4f}")
    elif "sar" in tag.lower():
        metrics["sar"].append(f"{segment},{value:.4f}")
    elif "g_loss" in tag.lower():
        metrics["g_loss"].append(f"{segment},{value:.4f}")
    elif "d_loss" in tag.lower():
        metrics["d_loss"].append(f"{segment},{value:.4f}")
    elif "val_loss" in tag.lower():
        metrics["val_loss"].append(f"{segment},{value:.4f}")

def write_epoch_summary(f, epoch, metrics, segment_metrics):
    f.write(f"Epoch {epoch + 1} Summary:\n")
    f.write(f"- Average SIR (Validation): {safe_mean(metrics['sir']):.4f}\n")
    f.write(f"- Average SDR (Validation): {safe_mean(metrics['sdr']):.4f}\n")
    f.write(f"- Average SAR (Validation): {safe_mean(metrics['sar']):.4f}\n")
    f.write(f"- Average Generator Loss (Training): {safe_mean(metrics['g_loss']):.4f}\n")
    f.write(f"- Average Discriminator Loss (Training): {safe_mean(metrics['d_loss']):.4f}\n")
    f.write(f"- Average Validation Loss: {safe_mean(metrics['val_loss']):.4f}\n\n")
    
    for key in segment_metrics:
        if segment_metrics[key]:
            f.write(f"Segment Metrics ({key.upper()}):\n")
            f.write("\n".join(segment_metrics[key]) + "\n\n")

def safe_mean(values):
    return sum(values) / len(values) if values else 0.0

# Example usage
if __name__ == "__main__":
    event_file = "path/to/event/file"
    output_file = "parsed_output.txt"
    parse_event_file(event_file, output_file, max_entries_per_tag=120)
