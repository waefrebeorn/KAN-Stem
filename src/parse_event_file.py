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
        for tag, entries in tag_data.items():
            for step, value in entries:
                epoch = step // 120  # Assuming 120 segments per epoch
                if current_epoch is None:
                    current_epoch = epoch
                if epoch != current_epoch:
                    # Write summary for the completed epoch
                    write_epoch_summary(f, current_epoch, epoch_metrics)
                    # Reset for the new epoch
                    current_epoch = epoch
                    epoch_metrics = {"sir": [], "sdr": [], "sar": [], "g_loss": [], "d_loss": [], "val_loss": []}
                # Write detailed segment metrics
                write_segment_metrics(f, step, tag, value)
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
        write_epoch_summary(f, current_epoch, epoch_metrics)

def write_epoch_summary(f, epoch, metrics):
    f.write(f"Epoch {epoch + 1} Summary:\n")
    f.write(f"- Average SIR (Validation): {sum(metrics['sir']) / len(metrics['sir']):.4f}\n")
    f.write(f"- Average SDR (Validation): {sum(metrics['sdr']) / len(metrics['sdr']):.4f}\n")
    f.write(f"- Average SAR (Validation): {sum(metrics['sar']) / len(metrics['sar']):.4f}\n")
    f.write(f"- Average Generator Loss (Training): {sum(metrics['g_loss']) / len(metrics['g_loss']):.4f}\n")
    f.write(f"- Average Discriminator Loss (Training): {sum(metrics['d_loss']) / len(metrics['d_loss']):.4f}\n")
    f.write(f"- Average Validation Loss: {sum(metrics['val_loss']) / len(metrics['val_loss']):.4f}\n\n")

def write_segment_metrics(f, step, tag, value):
    segment = step % 120 + 1
    f.write(f"Segment {segment}:\n")
    f.write(f"- {tag}: {value:.4f}\n")

# Example usage
if __name__ == "__main__":
    event_file = "path/to/event/file"
    output_file = "parsed_output.txt"
    parse_event_file(event_file, output_file, max_entries_per_tag=120)
