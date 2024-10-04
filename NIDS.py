import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance

import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df_train = pd.read_csv('datasets/UNSW_NB15_training-set.csv', header=0)
df_test = pd.read_csv('datasets/UNSW_NB15_testing-set.csv', header=0)

# Drop irrelevant features
df_train = df_train.drop(columns=['id'])
df_test = df_test.drop(columns=['id'])

# Encoding categorical variables (proto, service, state)
df_train = pd.get_dummies(df_train, columns=['proto', 'service', 'state'])
df_test = pd.get_dummies(df_test, columns=['proto', 'service', 'state'])

# Align columns between train and test datasets
X_train, X_test = df_train.align(df_test, join='left', axis=1, fill_value=0)

# Split the dataset into features and labels
y_train = df_train['label']
y_test = df_test['label']

# We want to predict the label, so we drop the label and attack_cat columns from the features b
X_train = X_train.drop(columns=['label', 'attack_cat'])
X_test = X_test.drop(columns=['label', 'attack_cat'])

# Drop state_CON
X_train = X_train.drop(columns=['state_CON'])


# Handle imbalanced data by applying class weights
# Ensure that classes are a numpy array, and calculate class weights
classes = np.array([0, 1])  # Define classes (0 for normal, 1 for attack)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights_dict = dict(zip(classes, class_weights))  # Convert to a dictionary


# Hyperparameter tuning with Optuna (Hyperparameters are already tuned and put in best_params)
"""
def objective(trial):
    # Define the search space for hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 10, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

    # Define the model with the hyperparameters
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   class_weight=class_weights_dict,
                                   random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Calculate F1 score
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    return f1


# Run Optuna for hyperparameter optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

# Use the best hyperparameters to train the final model
best_params = study.best_params
"""

best_params = {'n_estimators': 147, 'max_depth': 34, 'min_samples_split': 13, 'min_samples_leaf': 20}

# print(f"Best hyperparameters: {best_params}")

final_model = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                     max_depth=best_params['max_depth'],
                                     min_samples_split=best_params['min_samples_split'],
                                     min_samples_leaf=best_params['min_samples_leaf'],
                                     class_weight=class_weights_dict,
                                     random_state=42)

# Train the final model
final_model.fit(X_train, y_train)

# Test the final model
y_pred = final_model.predict(X_test)

# Evaluate the final model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Feature importance
features = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'proto_3pc', 'proto_a/n', 'proto_aes-sp3-d', 'proto_any', 'proto_argus', 'proto_aris', 'proto_arp', 'proto_ax.25', 'proto_bbn-rcc', 'proto_bna', 'proto_br-sat-mon', 'proto_cbt', 'proto_cftp', 'proto_chaos', 'proto_compaq-peer', 'proto_cphb', 'proto_cpnx', 'proto_crtp', 'proto_crudp', 'proto_dcn', 'proto_ddp', 'proto_ddx', 'proto_dgp', 'proto_egp', 'proto_eigrp', 'proto_emcon', 'proto_encap', 'proto_etherip', 'proto_fc', 'proto_fire', 'proto_ggp', 'proto_gmtp', 'proto_gre', 'proto_hmp', 'proto_i-nlsp', 'proto_iatp', 'proto_ib', 'proto_icmp', 'proto_idpr', 'proto_idpr-cmtp', 'proto_idrp', 'proto_ifmp', 'proto_igmp', 'proto_igp', 'proto_il', 'proto_ip', 'proto_ipcomp', 'proto_ipcv', 'proto_ipip', 'proto_iplt', 'proto_ipnip', 'proto_ippc', 'proto_ipv6', 'proto_ipv6-frag', 'proto_ipv6-no', 'proto_ipv6-opts', 'proto_ipv6-route', 'proto_ipx-n-ip', 'proto_irtp', 'proto_isis', 'proto_iso-ip', 'proto_iso-tp4', 'proto_kryptolan', 'proto_l2tp', 'proto_larp', 'proto_leaf-1', 'proto_leaf-2', 'proto_merit-inp', 'proto_mfe-nsp', 'proto_mhrp', 'proto_micp', 'proto_mobile', 'proto_mtp', 'proto_mux', 'proto_narp', 'proto_netblt', 'proto_nsfnet-igp', 'proto_nvp', 'proto_ospf', 'proto_pgm', 'proto_pim', 'proto_pipe', 'proto_pnni', 'proto_pri-enc', 'proto_prm', 'proto_ptp', 'proto_pup', 'proto_pvp', 'proto_qnx', 'proto_rdp', 'proto_rsvp', 'proto_rtp', 'proto_rvd', 'proto_sat-expak', 'proto_sat-mon', 'proto_sccopmce', 'proto_scps', 'proto_sctp', 'proto_sdrp', 'proto_secure-vmtp', 'proto_sep', 'proto_skip', 'proto_sm', 'proto_smp', 'proto_snp', 'proto_sprite-rpc', 'proto_sps', 'proto_srp', 'proto_st2', 'proto_stp', 'proto_sun-nd', 'proto_swipe', 'proto_tcf', 'proto_tcp', 'proto_tlsp', 'proto_tp++', 'proto_trunk-1', 'proto_trunk-2', 'proto_ttp', 'proto_udp', 'proto_unas', 'proto_uti', 'proto_vines', 'proto_visa', 'proto_vmtp', 'proto_vrrp', 'proto_wb-expak', 'proto_wb-mon', 'proto_wsn', 'proto_xnet', 'proto_xns-idp', 'proto_xtp', 'proto_zero', 'service_-', 'service_dhcp', 'service_dns', 'service_ftp', 'service_ftp-data', 'service_http', 'service_irc', 'service_pop3', 'service_radius', 'service_smtp', 'service_snmp', 'service_ssh', 'service_ssl', 'state_CON', 'state_ECO', 'state_FIN', 'state_INT', 'state_PAR', 'state_REQ', 'state_RST', 'state_URN', 'state_no']
result = permutation_importance(final_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
sorted_importances = sorted(zip(result.importances_mean, features), reverse=True)

for importance, feature in sorted_importances:
    print(f"{feature}: {importance}")

quit()


# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
