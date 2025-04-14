import os
import time
import multiprocessing
import numpy as np
import pandas as pd
import pyshark
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from queue import Queue
from threading import Thread
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

CONFIG = {
    'packet_capture_interface': 'Wi-Fi 2',
    'slow_model': LGBMClassifier(n_estimators=100),  # Only slow model is used
    'queue_size': 2000,
    'num_workers': 4,
    'packet_limit': 500,
    'confidence_threshold': 0.8 # Not used in single-model architecture
}

FEATURE_NAMES = [
    "Packet Size",
    "Protocol Type (TCP/UDP/ICMP)",
    "Time-to-Live (TTL)",
    "Source Port",
    "Destination Port",
    "Inter-arrival Time",
    "Direction (Inbound/Outbound)"
]

protocol_encoder = LabelEncoder()
protocol_encoder.fit(['TCP', 'UDP', 'ICMP'])

# Load and preprocess the UNSW_NB15 dataset
def load_and_preprocess_data(train_path, test_path):
    # Load training and testing datasets
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Drop unnecessary columns
    train_data = train_data.drop(columns=['id', 'attack_cat'])
    test_data = test_data.drop(columns=['id', 'attack_cat'])
    
    # Combine datasets to fit LabelEncoder on all possible categories
    combined_data = pd.concat([train_data, test_data], axis=0)
    
    # Encode categorical features
    categorical_cols = ['proto', 'service', 'state']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        combined_data[col] = le.fit_transform(combined_data[col])
        label_encoders[col] = le
    
    # Separate back into training and testing datasets
    train_data = combined_data[:len(train_data)]
    test_data = combined_data[len(train_data):]
    
    # Separate features and labels
    X_train = train_data.drop(columns=['label'])
    y_train = train_data['label']
    X_test = test_data.drop(columns=['label'])
    y_test = test_data['label']
    
    # Normalize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test, scaler

# Balance the dataset using SMOTE
def balance_data(X, y):
    print("Class distribution before balancing:", Counter(y))
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("Class distribution after balancing:", Counter(y_resampled))
    return X_resampled, y_resampled

# Extract features from a packet
def extract_features(packet, prev_timestamp):
    try:
        inter_arrival_time = time.time() - prev_timestamp if prev_timestamp else 0
        protocol = packet.transport_layer if hasattr(packet, 'transport_layer') else 'Unknown'
        protocol_encoded = protocol_encoder.transform([protocol])[0] if protocol in protocol_encoder.classes_ else -1
        
        src_ip = packet.ip.src if hasattr(packet, 'ip') else '0.0.0.0'
        dst_ip = packet.ip.dst if hasattr(packet, 'ip') else '0.0.0.0'
        src_port = int(packet.tcp.srcport) if hasattr(packet, 'tcp') else 0
        dst_port = int(packet.tcp.dstport) if hasattr(packet, 'tcp') else 0
        direction = 1 if src_ip.startswith(('192.', '10.', '172.')) else -1

        features = [
            int(packet.length),
            protocol_encoded,
            int(packet.ip.ttl) if hasattr(packet.ip, 'ttl') else 0,
            src_port,
            dst_port,
            inter_arrival_time,
            direction
        ]
        return {
            'flow_id': (src_ip, dst_ip, src_port, dst_port, protocol),
            'features': features,
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': src_port,
            'dst_port': dst_port,
            'protocol': protocol
        }
    except AttributeError:
        return None

# Capture packets from the network interface
def capture_packets(interface, limit):
    print(f"Starting packet capture on interface {interface}...")
    capture = pyshark.LiveCapture(interface=interface)
    packets = []
    prev_timestamp = None
    for i, packet in enumerate(capture.sniff_continuously(packet_count=limit)):
        features = extract_features(packet, prev_timestamp)
        prev_timestamp = time.time()
        if features:
            packets.append(features)
    return packets

# Process a packet using the slow model
def process_packet(packet, slow_model, stats):
    features = packet['features']
    prediction = slow_model.predict([features])[0]
    confidence = np.max(slow_model.predict_proba([features])[0])
    
    with stats['lock']:
        stats['slow_model_count'] += 1
        stats['confidences'].append(confidence)
    
    return prediction

# Worker function for processing packets
def worker(input_queue, output_queue, slow_model, stats):
    while True:
        packet = input_queue.get()
        if packet is None:
            break
        start_time = time.time()
        prediction = process_packet(packet, slow_model, stats)
        latency = time.time() - start_time
        with stats['lock']:
            stats['latencies'].append(latency)
            stats['processed_packets'] += 1
        output_queue.put((packet['flow_id'], prediction, packet['true_label']))  # Include true label

# Evaluate predictions
def evaluate_predictions(true_labels, predicted_labels):
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    accuracy = accuracy_score(true_labels, predicted_labels)
    return f1, accuracy

# Calculate 3-Packet Flow Miss Rate
def calculate_flow_miss_rate(packets):
    flow_counter = {}
    for packet in packets:
        flow_key = packet['flow_id']
        flow_counter[flow_key] = flow_counter.get(flow_key, 0) + 1
    
    missed_flows = sum(1 for count in flow_counter.values() if count < 3)
    total_flows = len(flow_counter)
    return missed_flows / total_flows if total_flows > 0 else 0

# Calculate metrics for live packet processing
def calculate_metrics(stats, packets, predicted_labels, true_labels):
    total_packets = stats['processed_packets']
    avg_latency = sum(stats['latencies']) / len(stats['latencies']) if stats['latencies'] else 0
    packets_per_second = total_packets / (stats['end_time'] - stats['start_time']) if stats['end_time'] > stats['start_time'] else 0
    avg_confidence = sum(stats['confidences']) / len(stats['confidences']) if stats['confidences'] else 0
    
    # Calculate 3-Packet Flow Miss Rate
    three_pkt_flow_miss_rate = calculate_flow_miss_rate(packets)
    
    # Calculate F1 score and accuracy
    f1, accuracy = evaluate_predictions(true_labels, predicted_labels)
    
    print(f"\n=== Metrics ===")
    print(f"Total Packets Processed: {total_packets}")
    print(f"Average Latency per Packet: {avg_latency:.6f} seconds")
    print(f"Packets Processed per Second: {packets_per_second:.2f}")
    print(f"Packets Processed by Slow Model: {stats['slow_model_count']}")
    print(f"Average Confidence of Slow Model: {avg_confidence:.2f}")
    print(f"Service Rate (Classifications Per Second): {packets_per_second:.2f}")
    print(f"Average End-to-End Latency (ms): {avg_latency * 1000:.2f}")
    print(f"3-Packet Flow Miss Rate: {three_pkt_flow_miss_rate:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

# Main function
def main():
    # Paths to the UNSW_NB15 dataset files
    train_path = r"Y:\BITS-Life\Sem-2\NS\Project\NIDS_Final\NTA_Basic\training.csv"
    test_path = r"Y:\BITS-Life\Sem-2\NS\Project\NIDS_Final\NTA_Basic\testing.csv"
    
    # Load and preprocess the dataset
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess_data(train_path, test_path)
    
    # Train the model on the first 7 features
    X_train_7 = X_train[:, :7]  # Use only the first 7 features
    X_test_7 = X_test[:, :7]
    
    slow_model = CONFIG['slow_model']
    slow_model.fit(X_train_7, y_train)
    
    # Evaluate the model on the testing data
    print("\nEvaluating Slow Model on Testing Data:")
    y_test_pred_slow = slow_model.predict(X_test_7)
    f1_slow, accuracy_slow = evaluate_predictions(y_test, y_test_pred_slow)
    print(f"Slow Model - F1 Score: {f1_slow:.4f}, Accuracy: {accuracy_slow:.4f}")
    
    # Replay the testing dataset as live packets
    print("\nReplaying testing dataset as live packets...")
    packets = []
    for i, row in enumerate(X_test_7):
        features = row  # Use the first 7 features
        packets.append({
            'flow_id': i,
            'features': features,
            'true_label': y_test.iloc[i]  # Use the actual label from the testing dataset
        })
    
    input_queue = Queue(CONFIG['queue_size'])
    output_queue = Queue()
    
    stats = {
        'latencies': [],
        'processed_packets': 0,
        'slow_model_count': 0,
        'confidences': [],
        'start_time': time.time(),
        'end_time': None,
        'lock': multiprocessing.Lock()
    }
    
    workers = [Thread(target=worker, args=(input_queue, output_queue, slow_model, stats)) for _ in range(CONFIG['num_workers'])]
    for t in workers:
        t.start()
    
    for packet in packets:
        input_queue.put(packet)
    
    # Wait for all packets to be processed
    for _ in range(CONFIG['num_workers']):
        input_queue.put(None)
    
    for t in workers:
        t.join()
    
    stats['end_time'] = time.time()
    
    # Collect results
    predicted_labels = []
    true_labels = []
    while not output_queue.empty():
        _, pred, true = output_queue.get()
        predicted_labels.append(pred)
        true_labels.append(true)
    
    # Calculate final metrics
    total_packets = stats['processed_packets']
    avg_latency = sum(stats['latencies']) / len(stats['latencies']) if stats['latencies'] else 0
    packets_per_second = total_packets / (stats['end_time'] - stats['start_time']) if stats['end_time'] > stats['start_time'] else 0
    avg_confidence = sum(stats['confidences']) / len(stats['confidences']) if stats['confidences'] else 0
    three_pkt_flow_miss_rate = calculate_flow_miss_rate(packets)
    f1, accuracy = evaluate_predictions(true_labels, predicted_labels)
    
    # Return results dictionary
    return {
        'total_packets': total_packets,
        'avg_latency': avg_latency,
        'packets_per_second': packets_per_second,
        'fast_model_count': 0,  # Slow model doesn't use fast model
        'slow_model_count': stats['slow_model_count'],
        'avg_confidence': avg_confidence,
        'flow_miss_rate': three_pkt_flow_miss_rate,
        'f1': f1,
        'accuracy': accuracy,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels
    }

if __name__ == "__main__":
    main()