import os, time
import numpy as np
import torch
import torch.nn as nn
from src.model import *
from src.util import *
import src.resnet as resnet, src.cbam as cbam

def inference(config):

    # Define Device
    device = config.device 
    # Data info
    data_keys = config.data_keys
    # Test config
    iter_log_ratio = config.iter_log_ratio

    # load test data
    dataset = config.dataset
    hdf5 = Hdf5(dataset=dataset, is_train=False, data_keys=data_keys, source_dir=config.path_source)
    source = hdf5.get_patchlist()
    with open(source, 'r') as f:
        test_filenames = np.array([line.strip() for line in f.readlines()])
        #np.random.shuffle(test_filenames)
    #NEW
    output_file = f"denoising_results_{config.timestamp}.txt" 

    # load model
    if config.path_model == "models/model_final_300_sm.pth":
        config.network = "cbam_test3_234"
    elif config.path_model == "models/model_final_100_sm.pth":
        config.network = "resnet_test2"
    network = config.network.lower()

    if network == "resnet_test2":
            res = resnet.resnet_test2
            print("network : resnet_test2")
    elif network == "cbam_test3_234":
        res = cbam.resnet_test3_234
        print("network : cbam_test3_234")
    else:
        raise ValueError("No resnet match {}".format(network))

    net = {}
    net["sm"] = nn.DataParallel(Resnet_sm(res=res).to(device))#.to(device)
    net["sm"].load_state_dict(torch.load(config.path_model, map_location=device))
    def set_bn_eval(m):
        if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
            m.eval()

    for k in net.keys():
        for name, param in net[k].named_parameters():
            param.requires_grad = False
        net[k].apply(set_bn_eval)

    # testing
    print("Device :", device)
    print("Testing source : ", source)
    print("Testing...")
    d = ["sm"] 
    eval_list = ['acc', 'sen', 'spec']
    step_info = {key2: {key:[] for key in ["sm"]} for key2 in eval_list} #NEW

    step_size = len(test_filenames)
    step_log = int(step_size * iter_log_ratio)
    start_time = time.time()

    out_stack ={key:[] for key in ["sm"]} # To store the labels (0 or 1) of the predicted classes
    prob_stack = {key: [] for key in ["sm"]} # To store the probabilities of the predicted classes
    label_stack = []
    with torch.no_grad():
        for step in range(step_size):
            # load batch data
            train_batch_dic = hdf5.getDataDicByName(test_filenames[step])
            data = torch.tensor(train_batch_dic['data'], dtype=torch.float).to(device)
            _k = 'tdata'
            tdata = torch.tensor(train_batch_dic[_k], dtype=torch.float).to(device)
            if config.if_labels:
                label = torch.tensor(train_batch_dic['label'].squeeze(), dtype=torch.long).to(device)
            tdata = tdata.view(-1, 1, tdata.shape[-1])

            # forward
            out = {}
            out["sm"], _ = net["sm"](data)

            # Collect probabilities (softmax output)
            softmax_prob = torch.nn.functional.softmax(out["sm"], dim=1)

            # loss
            for key in net.keys():
                out[key] = out[key].data

            if config.if_labels:
                label_stack.append(label.cpu().detach().tolist())

            for key in out.keys():
                #out_stack[key].extend(np.argmax(out[key].cpu().detach().tolist(), axis=1))

                # Collect probability of the predicted class
                predictions = torch.argmax(out[key], dim=1)
                predicted_probs = softmax_prob.gather(1, predictions.unsqueeze(1)).squeeze(1)
                prob_stack[key].extend(predicted_probs.cpu().detach().tolist())

                adjusted_labels = []
                for prob, pred in zip(predicted_probs, predictions):
                    if prob < config.probability and pred.item() == 1:
                        new_label = 0
                    else:
                        new_label = pred.item()

                    adjusted_labels.append(new_label)

                out_stack[key].extend(adjusted_labels)

                if config.if_labels:
                    step_info['acc'][key] = accuracy(out_stack[key], label_stack)
                    step_info['sen'][key] = sensitivity(out_stack[key], label_stack)
                    step_info['spec'][key] = specificity(out_stack[key], label_stack)

            if config.if_labels and step_log !=0 and (step + 1) % step_log == 0:
                log = "step [{:3}/{}] time [{:.1f}s]".format(step + 1, step_size, time.time() - start_time) + \
                " | sm acc [{:.3f}] sen [{:.3f}] spec [{:.3f}]".format(*[step_info[key]["sm"] for key in eval_list])
                print(log)
            elif step_log !=0 and (step + 1) % step_log == 0:
                log = "step [{:3}/{}] time [{:.1f}s]".format(step + 1, step_size, time.time() - start_time) 
                print(log)

    
    output_file_label_1 = f"signal_comp_{config.timestamp}.txt"

    with open(output_file, 'a') as f_all, open(output_file_label_1, 'a') as f_label_1:
        for filename, label, prob in zip(test_filenames, out_stack["sm"], prob_stack["sm"]):
            prob_percent = prob * 100
            line = f"{filename} Class: {label} Probability: {prob_percent:.2f}%\n"
            
            f_all.write(line)
            
            if label == 1:
                f_label_1.write(line)
    print("[!] Testing complete. Info is saved in", output_file)
    print("[!] Info for signal data is saved in", output_file_label_1)