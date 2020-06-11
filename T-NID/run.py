import argparse
import copy
import os
import time
from itertools import combinations, chain
import torch
import torch.nn as nn
import numpy as np
from neural_interaction_detection import get_interactions
from multilayer_perceptron import MLP, train, get_weights
from utils import preprocess_data, get_auc, get_anyorder_R_precision, set_seed, print_rankings
from scipy.stats import mode as stats_mode
from neural_interaction_detection import prune_redundant_interactions

if os.path.exists("/Users/sam/Documents/Programming/Research"):
    slurm = False
else:
    slurm = True


def print_latex_tables(row_names, tables, formatted_means, formatted_stds, line_marker="|"):
    col_index = 0
    running_latex = ""
    for col_names in tables[:-1]:
        latex = r"""\begin{center}
                    \begin{tabular}{l|""" + "".join("c|" if line_marker in c else "c" for c in col_names) + """}
                    \hline & """ + " & ".join(col_names) + r""" \\
                    \hline"""
        for row, row_name in enumerate(row_names):
            latex += row_name + " & " + " & ".join(
                ["${} \pm {}$".format(item[0], item[1]) for item in zip(formatted_means[row].tolist()[col_index:col_index + len(col_names)], formatted_stds[row].tolist()[col_index:col_index + len(col_names)])]) + r""" \\
            """
        latex += r"""\hline average""" + " & " + " & ".join(["${} \pm {}$".format(item[0], item[1]) for item in
                                                             zip(np.around(np.mean(formatted_means, 0), decimals=3).tolist()[
                                                                 col_index:col_index + len(col_names)],
                                                                 np.around(np.mean(formatted_stds, 0), decimals=3).tolist()[
                                                             col_index:col_index + len(col_names)])]) + r""" \\
                    \hline
                    \end{tabular}
                    \end{center}"""

        print(latex)
        col_index += len(col_names)
        running_latex += latex

    latex = r"""\begin{center}
                \begin{tabular}{l|""" + "".join("c|" if line_marker in c else "c" for c in tables[-1]) + """}
                \hline & """ + " & ".join(tables[-1]) + r""" \\
                \hline"""
    for row, row_name in enumerate(row_names):
        latex += row_name + " & " + " & ".join(["${} \pm {}$".format(item[0], item[1]) for item in zip(formatted_means[row].tolist()[col_index:], formatted_stds[row].tolist()[col_index:])]) + r""" \\
        """
    latex += r"""\hline average""" + " & " + " & ".join(
        ["${} \pm {}$".format(item[0], item[1]) for item in zip(np.around(np.mean(formatted_means, 0), decimals=3).tolist()[col_index:], np.around(np.mean(formatted_stds, 0), decimals=3).tolist()[col_index:])]) + r""" \\
                \hline
                \end{tabular}
                \end{center}"""

    print(latex)
    running_latex += latex
    return running_latex


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


def f_1(x):
    return np.pi ** (x[0] * x[1]) * np.sqrt(2 * x[2]) - (np.arcsin(x[3])) + np.log(x[2] + x[4]) - \
           (x[8] / x[9]) * np.sqrt(x[6] / x[7]) - x[1] * x[6], [{0, 1, 2}, {2, 4}, {6, 7, 8, 9}, {1, 6}]


def f_2(x):
    return np.pi ** (x[0] * x[1]) * np.sqrt(2 * np.abs(x[2])) - (np.arcsin(0.5 * x[3])) + np.log(
        np.abs(x[2] + x[4]) + 1) - \
           (x[8] / (1 + np.abs(x[9]))) * np.sqrt(np.abs(x[6]) / (1 + np.abs(x[7]))) - x[1] * x[6], [{0, 1, 2}, {2, 4},
                                                                                                    {6, 7, 8, 9},
                                                                                                    {1, 6}]


def f_3(x):
    return np.exp(np.abs(x[0] - x[1])) + np.abs(x[1] * x[2]) - ((x[2]) ** 2) ** np.absolute(x[3]) + \
           np.log(x[3] ** 2 + x[4] ** 2 + x[6] ** 2 + x[7] ** 2) + x[8] + 1. / (1 + x[9] ** 2), \
           [{0, 1}, {1, 2}, {2, 3}, {3, 4, 6, 7}]


def f_4(x):
    return np.exp(np.abs(x[0] - x[1])) + np.abs(x[1] * x[2]) - ((x[2]) ** 2) ** np.absolute(x[3]) + (x[0] * x[3]) ** 2 \
           + np.log(x[3] ** 2 + x[4] ** 2 + x[6] ** 2 + x[7] ** 2) + x[8] + 1. / (1 + x[9] ** 2), [{0, 1}, {1, 2},
                                                                                                   {2, 3}, {0, 3},
                                                                                                   {3, 4, 6, 7}]


def f_5(x):
    return 1. / (1 + x[0] ** 2 + x[1] ** 2 + x[2] ** 2) + np.sqrt(np.abs(x[3] + x[4])) + np.abs(x[5] + x[6]) + x[7] * x[
        8] * \
           x[9], [{0, 1, 2}, {3, 4}, {5, 6}, {7, 8, 9}]


def f_6(x):
    return np.exp(np.abs(x[0] * x[1]) + 1) - np.exp(np.abs(x[2] + x[3]) + 1) + np.cos(x[4] + x[5] - x[7]) + np.sqrt(
        x[7] ** 2 + x[8] ** 2 + x[9] ** 2), [{0, 1}, {2, 3}, {4, 5, 7}, {7, 8, 9}]


def f_7(x):
    return (np.arctan(x[0]) + np.arctan(x[1])) ** 2 + np.max((x[2] * x[3] + x[5], np.zeros(x.shape[1])), 0) - \
           (1 + (x[3] * x[4] * x[5] * x[6] * x[7])) ** -2 + (np.abs(x[6]) / (1 + np.abs(x[8]))) ** 5 + np.sum(x, 0), \
           [{0, 1}, {2, 3, 5}, {3, 4, 5, 6, 7}, {6, 8}]


def f_8(x):
    return x[0] * x[1] + 2 ** (x[2] + x[4] + x[5]) + 2 ** (x[2] + x[3] + x[4] + x[6]) + np.sin(
        x[6] * np.sin(x[7] + x[8])) + np.arccos(0.9 * x[9]), [{0, 1}, {2, 4, 5}, {2, 3, 4, 6}, {6, 7, 8}]


def f_9(x):
    return np.tanh(x[0] * x[1] + x[2] * x[3]) * np.sqrt(np.abs(x[4])) + np.exp(x[4] + x[5]) + np.log(
        (x[5] * x[6] * x[7]) ** 2 + 1) + x[8] * x[9] + 1. / (1 + np.abs(x[9])), [{0, 1, 2, 3, 4}, {4, 5}, {5, 6, 7},
                                                                                 {8, 9}]


def f_10(x):
    return np.sinh(x[0] + x[1]) + np.arccos(np.tanh(x[2] + x[4] + x[6])) + np.cos(x[3] + x[4]) + 1. / np.cos(
        x[6] * x[8]), [{0, 1}, {2, 4, 6}, {3, 4}, {6, 8}]


def generate_X(size, num):
    if num > 1:
        return np.random.uniform(low=-1, high=1, size=(size, 10))
    else:
        x4 = np.random.uniform(low=0.6, high=1, size=(size, 1))
        x5 = np.random.uniform(low=0.6, high=1, size=(size, 1))
        x8 = np.random.uniform(low=0.6, high=1, size=(size, 1))
        x10 = np.random.uniform(low=0.6, high=1, size=(size, 1))
        x1 = np.random.uniform(low=0, high=1, size=(size, 1))
        x2 = np.random.uniform(low=0, high=1, size=(size, 1))
        x3 = np.random.uniform(low=0, high=1, size=(size, 1))
        x6 = np.random.uniform(low=0, high=1, size=(size, 1))
        x7 = np.random.uniform(low=0, high=1, size=(size, 1))
        x9 = np.random.uniform(low=0, high=1, size=(size, 1))
        return np.concatenate([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], 1)


def get_representative_sample(data, mode="closest_to_median", order=2):
    data = np.array(data)
    if "closest_to_pairwise_reduced_" in mode:
        sample = np.array(stats_mode(get_representative_sample(data, mode.replace("reduced_", ""))))[0, 0]
    elif "closest_to_pairwise_" in mode:  # Can also try pairwise distance from the global aggregate
        sample = []
        interactions = [list(combinations(np.arange(data.shape[1]), o)) for o in
                        range(2, order + 1)]
        for pair in combinations(np.arange(data.shape[1]), order):
            if "median" in mode:
                closest_to = np.median(data[:, pair], 0)
            elif "mean" in mode:
                closest_to = np.mean(data[:, pair], 0)
            elif "min" in mode:
                closest_to = np.min(data[:, pair], 0)
            elif "max" in mode:
                closest_to = np.max(data[:, pair], 0)
            elif "mode" in mode:
                closest_to = np.array(stats_mode(data[:, pair]))[0, 0]
            distances = np.linalg.norm(data[:, pair] - np.tile(closest_to, (data.shape[0], 1)), axis=1)
            sample.append(data[np.argmin(distances)])
        sample = np.array(sample)
    elif "closest_to_" in mode:  # Might be better to pass in the aggregate
        if "median" in mode:
            closest_to = np.median(data, 0)
        elif "mean" in mode:
            closest_to = np.mean(data, 0)
        elif "min" in mode:
            closest_to = np.min(data, 0)
        elif "max" in mode:
            closest_to = np.max(data, 0)
        elif "mode" in mode:
            closest_to = np.array(stats_mode(data))[0, 0]
        distances = np.linalg.norm(data - np.tile(closest_to, (data.shape[0], 1)), axis=1)
        sample = data[np.argmin(distances)]

    return sample


def pairwise_reduce(interaction_effects, inter, interaction):
    return interaction_effects[inter]


def jacobian(x, y=None, model=None, refs=None, k=None, device=torch.device("cpu"), create_graph=False, abs_val=True, greedy_heuristic=False):
    # Require x to be list of batch-of-1 items (s.t. they can be fed into model individually w/o sub-indexing)
    assert isinstance(x, (list, tuple))
    jac = []
    new_refs = []

    if refs is None and y is not None:
        refs = [[] for _ in range(len(x))]
    for b in range(len(x)):
        st = time.time()
        assert torch.is_tensor(x[b])
        if y is None:
            # Require the extra empty batch dimension for feeding into model
            assert len(x[b].shape) == 2
        x[b].to(device)
        x[b].requires_grad = True
        # Num classes x 1 or num x x.shape[2]
        y_b = model.eval()(x[b]) if y is None else y[b]
        jac_b = []
        new_refs_b = []
        non_redundant_refs = set()
        valid_inds = []
        entity_saliences = {}
        interaction_saliences = []
        if y is None:
            valid_inds.append((0, 0))
        else:
            for c in range(len(y_b)):
                for p in range(len(y_b[c])):
                    # Check non-redundant
                    if len(refs[b]) > 0:
                        ref = sorted(refs[b][c].tolist() + [p])
                        if str(ref) in non_redundant_refs or len(set(ref)) < len(ref):
                            continue
                        non_redundant_refs.add(str(ref))
                    valid_inds.append((c, p))
                    # Consider for top k
                    if k is not None:
                        if len(refs[b]) == 0:
                            refs_b_c = []
                        else:
                            refs_b_c = refs[b][c].tolist()
                        if greedy_heuristic:
                            interaction_saliences.append(torch.abs(y_b[c][p]) if abs_val else y_b[c][p])
                        for ind in refs_b_c + [p]:
                            if ind not in entity_saliences:
                                entity_saliences[ind] = 0
                            if abs_val:
                                entity_saliences[ind] += torch.abs(y_b[c][p])
                            else:
                                entity_saliences[ind] += y_b[c][p]
            if k is not None:
                if len(interaction_saliences) == 0:
                    for c, p in valid_inds:
                        salience = 0
                        if len(refs[b]) == 0:
                            refs_b_c = []
                        else:
                            refs_b_c = refs[b][c].tolist()
                        for ind in refs_b_c + [p]:
                            salience += entity_saliences[ind]
                        interaction_saliences.append(salience)
                valid_inds = [inds for _, inds in sorted(zip(interaction_saliences, valid_inds), key=lambda si: si[0])][
                             -k:]
        # if b == 0:
        #     print("Batch {}, Time of Processing: {}".format(b, time.time() - st))
        stt = time.time()
        for c, p in valid_inds:
            st = time.time()
            jac_b.append(torch.autograd.grad(y_b[c][p], x[b], retain_graph=True, create_graph=create_graph)[0][0])
            # if b == 0:
            #     print("Valid indices: {}, {}, Time of Gradient: {}".format(c, p, time.time() - st))
            if y is not None:
                if len(refs[b]) == 0:
                    new_refs_b.append(torch.tensor([p]))
                else:
                    new_refs_b.append(torch.sort(torch.cat((refs[b][c], torch.tensor([p])), -1))[0])
        # if b == 0:
        #     print("Total time of gradient computations: ", time.time() - stt)

        jac.append(jac_b)
        new_refs.append(new_refs_b)
    # Batch size x num classes x x.shape[2], batch size x num classes x order - 1
    return jac, new_refs


def compute_interactions_and_auc(inputs, model, ground_truth, order=2, o=2, interaction_effects=None,
                                 device=torch.device("cpu"), mode=None, interactions=None, abs_val=True, verbose=False, greedy_heuristic=False):
    start = time.time()
    if interactions is None:
        interactions = set()
    else:
        interactions = set(interactions)

    # Compute cross derivatives
    if interaction_effects is None:
        inputs = torch.FloatTensor(np.array(inputs)) if isinstance(inputs, list) else torch.FloatTensor(inputs)
        if len(inputs.shape) == 1:
            inputs = [inputs.unsqueeze(0)]
        elif len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(1).unbind()
        oth_order = [jacobian(inputs, model=model, device=device, create_graph=True, abs_val=abs_val, greedy_heuristic=greedy_heuristic)[0]]
        oth_ref = [None]
        for o_ in range(2, order + 1):
            if verbose:
                print("Order: ", o_)
            if o_ <= o:
                oth_o, oth_r = jacobian(inputs, oth_order[-1], create_graph=o_ < order, device=device, refs=oth_ref[-1], abs_val=abs_val, greedy_heuristic=greedy_heuristic)
                oth_order.append(oth_o)
                oth_ref.append(oth_r)
            else:
                oth_o, oth_r = jacobian(inputs, oth_order[-1], create_graph=o_ < order, device=device, refs=oth_ref[-1],
                                        k=inputs[0].shape[1] ** (o - 1), abs_val=abs_val, greedy_heuristic=greedy_heuristic)
                oth_order.append(oth_o)
                oth_ref.append(oth_r)
        # Per order: batches x num x input size, Per order: batches x num x order - 1
        for o in range(1, len(oth_order)):
            for b in range(len(oth_order[o])):
                oth_order[o][b] = torch.stack(oth_order[o][b])
                oth_ref[o][b] = torch.stack(oth_ref[o][b])
                if abs_val:
                    oth_order[o][b] = torch.abs(oth_order[o][b])
        oth_order = oth_order[1:]
        oth_ref = oth_ref[1:]
        if verbose:
            print("Time of cross derivative computation: ", time.time() - start)

    # Aggregate across different cross derivatives according to mode
    AUCs = []
    r_precs = []
    if interaction_effects is None:
        interaction_effects = [[] for _ in oth_order[0]]
    mode_interaction_effects = {}
    mode_add_interactions = copy.deepcopy(interactions)
    for b in range(len(interaction_effects)):
        add_interactions = copy.deepcopy(interactions)
        if len(interaction_effects[b]) == 0:
            for o, oth in enumerate(oth_order):
                non_redundant_interactions = set()
                for c in range(oth_order[o][b].shape[0]):
                    for p in range(inputs[0].shape[1]):
                        interaction = tuple(sorted(oth_ref[o][b][c].tolist() + [p]))
                        if interaction in non_redundant_interactions or len(set(interaction)) < len(interaction):
                            continue
                        non_redundant_interactions.add(interaction)
                        if mode is None:
                            interaction_effects[b].append((interaction, oth_order[o][b][c][p].item()))
                            if interaction in add_interactions:
                                add_interactions.remove(interaction)
                        else:
                            if interaction in mode_interaction_effects:
                                mode_interaction_effects[interaction].append(oth_order[o][b][c][p].item())
                            else:
                                mode_interaction_effects[interaction] = [oth_order[o][b][c][p].item()]
                                if interaction in mode_add_interactions:
                                    mode_add_interactions.remove(interaction)
        else:
            if mode is not None:
                for interaction_effect in interaction_effects[b]:
                    if interaction_effect[0] in mode_interaction_effects:
                        mode_interaction_effects[interaction_effect[0]].append(interaction_effect[1])
                    else:
                        mode_interaction_effects[interaction_effect[0]] = [interaction_effect[1]]
                        if interaction_effect[0] in mode_add_interactions:
                            mode_add_interactions.remove(interaction_effect[0])

        if mode is None:
            for interaction in add_interactions:
                interaction_effects[b].append((tuple(interaction), 0))
            AUC = [get_auc([item for item in interaction_effects[b] if len(item[0]) == oth + 1],
                           ground_truth, verbose=verbose)
                   if max([len(g) for g in ground_truth]) >= oth + 1 else 0 for oth in range(1, order)]
            AUCs.append(AUC)
            interaction_effects_n_pruned = prune_redundant_interactions(interaction_effects[b])
            r_prec = get_anyorder_R_precision(interaction_effects_n_pruned, ground_truth)
            r_precs.append(r_prec)
            if verbose:
                print("{}th sample AUC: {}".format(b, AUC))
    if mode is not None:
        for interaction in mode_add_interactions:
            mode_interaction_effects[tuple(interaction)] = [0]
        if not isinstance(mode, list):
            mode = [mode]
        interaction_effects = []
        for mth, m in enumerate(mode):
            if abs_val is None:
                interaction_effects_m = [(interaction, m(mode_interaction_effects[interaction],
                                                                inter, interaction) if m == pairwise_reduce
                else m(mode_interaction_effects[interaction])[0][0] if m == stats_mode
                else m(mode_interaction_effects[interaction]))
                                         for inter, interaction in enumerate(mode_interaction_effects)]
            else:
                interaction_effects_m = [(interaction, np.abs(m(mode_interaction_effects[interaction],
                                                                inter, interaction)) if m == pairwise_reduce
                                         else np.abs(m(mode_interaction_effects[interaction]))[0][0] if m == stats_mode
                                         else np.abs(m(mode_interaction_effects[interaction])))
                                         for inter, interaction in enumerate(mode_interaction_effects)]
            AUC = [get_auc([item for item in interaction_effects_m if len(item[0]) == oth + 1],
                           ground_truth, verbose=verbose)
                   if max([len(g) for g in ground_truth]) >= oth + 1 else 0 for oth in range(1, order)]
            AUCs.append(AUC)
            interaction_effects_m_pruned = prune_redundant_interactions(interaction_effects_m)
            r_prec = get_anyorder_R_precision(interaction_effects_m_pruned, ground_truth)
            r_precs.append(r_prec)
            interaction_effects.append(interaction_effects_m)
            if verbose:
                print("{}th sample AUC: {}".format(mth, AUC))
    return AUCs, interaction_effects


def test_inputs_n_way(X_train, model, ground_truth, device, abs_val, order, o, interactions, verbose, greedy_heuristic):
    representative_samples = [get_representative_sample(X_train, mode="closest_to_{}".format(mode))
                              for mode in ["mean", "median", "min", "max", "mode"]]
    representative_samples.append(X_train[np.random.randint(X_train.shape[0])])
    sample_interaction_effects = compute_interactions_and_auc(inputs=representative_samples, model=model,
                                                              ground_truth=ground_truth, o=o, device=device,
                                                              abs_val=abs_val, order=order, verbose=verbose)[1]

    aucs = []
    interaction_effects = []
    for representatives in list(powerset(sample_interaction_effects)):
        for mode in [np.mean, np.median, np.min, np.max, stats_mode]:
            auc, inters = compute_interactions_and_auc(inputs=representatives, mode=mode, model=model,
                                                       ground_truth=ground_truth, o=o, device=device, abs_val=abs_val,
                                                       order=order, verbose=verbose, interactions=interactions,
                                                       interaction_effects=representatives,
                                                       greedy_heuristic=greedy_heuristic)
            aucs += auc
            interaction_effects += inters

    return aucs, interaction_effects


def run(num, epochs=33, layer_sizes=None, activation=nn.ELU, use_main_effect_nets=True, num_samples=30000,
        num_features=10, valid_size=5, test_size=5, std_scale=True, my_data_norm=False, lr=1e-3, l1_const=5e-5,
        dropout_p=0, early_stopping=False, patience=5, abs_val=True, verbose=True, order=2, o=2,
        greedy_heuristic=False, gelu_final_layer=False, gelu_last_layer=False, gelu_alt_layer=False,
        gelu_main_effects=False):
    # Params
    device = torch.device("cuda" if args.cuda else "cpu")
    if layer_sizes is None:
        layer_sizes = [140, 100, 60, 20]

    # Data
    # set_seed(42)
    X = generate_X(num_samples, num)
    Y, ground_truth = globals()["f_{}".format(num)](X.transpose())
    if my_data_norm:
        X = np.array(X)
        X = (X - X.min(0)) / X.ptp(0)
    data_loaders = preprocess_data(X, Y, valid_size=valid_size, test_size=test_size, std_scale=std_scale,
                                   get_torch_loaders=True)
    X_train = np.concatenate([data[0] for data in data_loaders["train"]], 0)

    # Model and training
    model = MLP(num_features, layer_sizes, use_main_effect_nets=use_main_effect_nets, activation=activation,
                dropout_p=dropout_p, gelu_final_layer=gelu_final_layer, gelu_last_layer=gelu_last_layer,
                gelu_alt_layer=gelu_alt_layer, gelu_main_effects=gelu_main_effects).to(device)
    model, mlp_loss = train(model, data_loaders, nepochs=epochs, device=device, learning_rate=lr, l1_const=l1_const,
                            verbose=verbose, early_stopping=early_stopping, patience=patience)

    # NID AUC
    model_weights = get_weights(model)
    pairwise_interactions, _ = get_interactions(model_weights, pairwise=True, one_indexed=True)
    # Automatically selects the top 100 excluding redundant subsets, and unpruned -- can use internal func to prune mine
    anyorder_interactions_pruned, anyorder_interactions_unpruned = get_interactions(model_weights, one_indexed=True)
    anyorder_interactions_unpruned = [inter for inter in anyorder_interactions_unpruned if len(inter[0]) <= order]
    # auc_nid = get_auc(pairwise_interactions, [{i + 1 for i in inter} for inter in ground_truth], verbose=verbose)
    if order == 2:
        anyorder_interactions_unpruned = pairwise_interactions
    # My AUC
    n_way_NID = set(
        [tuple([inr - 1 for inr in inter[0]]) for inter in anyorder_interactions_unpruned if len(inter[0]) <= order])
    auc_mine, interactions = test_inputs_n_way(X_train, model, ground_truth, device, abs_val, order, o, n_way_NID,
                                         verbose, greedy_heuristic=greedy_heuristic)
    # auc_mine = aucs1

    aucs_nid = []
    for nth in interactions:
        new = copy.deepcopy(anyorder_interactions_unpruned)
        for interaction in [inr[0] for inr in nth]:
            if interaction not in n_way_NID:
                new.append(((inr + 1 for inr in interaction), 0))

        # two_and_three_way = [inter for inter in anyorder_interactions_unpruned if len(inter[0]) <= order]
        # print(set([tuple([inr - 1 for inr in inter[0]]) for inter in new]) == set([inr[0] for inr in nth]))
        # print([tuple([inr - 1 for inr in inter[0]]) for inter in new])
        # print([inr[0] for inr in nth])
        auc_nid = [get_auc([item for item in list(new) if len(tuple(tuple(item)[0])) == oth + 1] if oth + 1 > 2
                           else pairwise_interactions,
                           [{i + 1 for i in inter} for inter in ground_truth], verbose=verbose)
                   if max([len(g) for g in ground_truth]) >= oth + 1 else 0
                   for oth in range(1, order)]
        aucs_nid.append(auc_nid)

    # Requires a subset of "detected" higher-order interactions and computes precision (% of those are real)
    r_prec = get_anyorder_R_precision(anyorder_interactions_pruned, [{i + 1 for i in inter} for inter in ground_truth])

    return auc_mine, aucs_nid


# Command line params
parser = argparse.ArgumentParser(description='Interaction detection')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='Number of epochs to train (default: 100)')
parser.add_argument('--layer_sizes', nargs='*', default=[140, 100, 60, 20], required=False, type=int,
                    help='Number of hidden units for each layer of MLP e.g. "140 100 60 20"')
parser.add_argument('--activation', type=str, default="GELU", help='Name of activation function e.g. "GELU"')
parser.add_argument('--disable_main_effect_nets', action='store_true', default=False,
                    help='Whether to disable the main effects univariate networks')
parser.add_argument('--num_samples', type=int, default=30000, help='Number of samples to generate')
parser.add_argument('--valid_size', type=int, default=10000, help='Number of validation samples to use')
parser.add_argument('--test_size', type=int, default=10000, help='Number of test samples to use')
parser.add_argument('--disable_std_scale', action='store_true', default=False,
                    help='Whether to disable normalizing by std')
parser.add_argument('--my_data_norm', action='store_true', default=False, help='Whether to normalize to [0, 1]')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--l1_const', type=float, default=5e-5, help='L1 regularization constant')
parser.add_argument('--dropout_p', type=float, default=0, help='Dropout prob')
parser.add_argument('--disable_early_stopping', action='store_true', default=False,
                    help='Whether to disable early stopping')
parser.add_argument('--patience', type=int, default=100, help='Patience steps for early stopping')
parser.add_argument('--disable_abs_val', action='store_true', default=False,
                    help='Whether to disable absolute value of each hessian')
parser.add_argument('--verbose', action='store_true', default=False, help='Print outputs')
parser.add_argument('--order', type=int, default=2, help='Compute up to N-way interactions')
parser.add_argument('--o', type=int, default=2, help='Use subsampling heuristic after this point')
parser.add_argument('--greedy_heuristic', action='store_true', default=False,
                    help='Whether to use greedy heuristic')
parser.add_argument('--use_default_param_set', type=int, default=14,
                    help='Number >0 (1-indexed) of predefined param set to enable; if 0, then use args')
parser.add_argument('--num_trials', type=int, default=10, help='Number of trials to average over')
args = parser.parse_args()
args.cuda = slurm and torch.cuda.is_available()

# Set params
if args.use_default_param_set == 0:
    params = dict(epochs=args.epochs, layer_sizes=args.layer_sizes, activation=getattr(nn, args.activation),
                  use_main_effect_nets=not args.disable_main_effect_nets, num_samples=args.num_samples,
                  valid_size=args.valid_size, test_size=args.test_size, std_scale=not args.disable_std_scale,
                  my_data_norm=args.my_data_norm, lr=args.lr, l1_const=args.l1_const, dropout_p=args.dropout_p,
                  early_stopping=not args.disable_early_stopping, patience=args.patience,
                  abs_val=not args.disable_abs_val, order=args.order, verbose=args.verbose, o=args.o,
                  greedy_heuristic=args.greedy_heuristic)
else:
    class Gelu(nn.Module):
        def __init__(self):
            super(Gelu, self).__init__()

        def forward(self, x):
            return 0.5 * x * (1 + torch.tanh(np.math.sqrt(2 / np.math.pi) * (x + 0.044715 * torch.pow(x, 3))))


    params = [
        # 1 = normal greedy
        dict(epochs=200, layer_sizes=[140, 100, 60, 20], activation=nn.GELU, use_main_effect_nets=True,
             num_samples=30000, valid_size=10000, test_size=10000, std_scale=True,
             my_data_norm=False, lr=1e-3, l1_const=5e-5, dropout_p=0, early_stopping=True, patience=10,
             abs_val=True, verbose=False, order=5, o=2, greedy_heuristic=True),
        # 2 = extra layer 180
        dict(epochs=200, layer_sizes=[180, 140, 100, 60, 20], activation=nn.GELU, use_main_effect_nets=True,
             num_samples=30000, valid_size=10000, test_size=10000, std_scale=True,
             my_data_norm=False, lr=1e-3, l1_const=5e-5, dropout_p=0, early_stopping=True, patience=10,
             abs_val=True, verbose=False, order=5, o=2, greedy_heuristic=True),
        # 3 = no main effects
        dict(epochs=200, layer_sizes=[140, 100, 60, 20], activation=nn.GELU, use_main_effect_nets=False,
             num_samples=30000, valid_size=10000, test_size=10000, std_scale=True,
             my_data_norm=False, lr=1e-3, l1_const=5e-5, dropout_p=0, early_stopping=True, patience=10,
             abs_val=True, verbose=False, order=5, o=2, greedy_heuristic=True),
        # 4 = lr 5e3
        dict(epochs=200, layer_sizes=[140, 100, 60, 20], activation=nn.GELU, use_main_effect_nets=True,
             num_samples=30000, valid_size=10000, test_size=10000, std_scale=True,
             my_data_norm=False, lr=5e-3, l1_const=5e-5, dropout_p=0, early_stopping=True, patience=10,
             abs_val=True, verbose=False, order=5, o=2, greedy_heuristic=True),
        # 5 = ReLU
        dict(epochs=200, layer_sizes=[140, 100, 60, 20], activation=nn.ReLU, use_main_effect_nets=True,
             num_samples=30000, valid_size=10000, test_size=10000, std_scale=True,
             my_data_norm=False, lr=1e-3, l1_const=5e-5, dropout_p=0, early_stopping=True, patience=10,
             abs_val=True, verbose=False, order=5, o=2, greedy_heuristic=True),
        # 6 = ReLU with final layer
        dict(epochs=200, layer_sizes=[140, 100, 60, 20], activation=nn.ReLU, use_main_effect_nets=True,
             num_samples=30000, valid_size=10000, test_size=10000, std_scale=True,
             my_data_norm=False, lr=1e-3, l1_const=5e-5, dropout_p=0, early_stopping=True, patience=10,
             abs_val=True, verbose=False, order=5, o=2, greedy_heuristic=True, gelu_final_layer=True),
        # 7 = ReLU with last layer
        dict(epochs=200, layer_sizes=[140, 100, 60, 20], activation=nn.ReLU, use_main_effect_nets=True,
             num_samples=30000, valid_size=10000, test_size=10000, std_scale=True,
             my_data_norm=False, lr=1e-3, l1_const=5e-5, dropout_p=0, early_stopping=True, patience=10,
             abs_val=True, verbose=False, order=5, o=2, greedy_heuristic=True, gelu_last_layer=True),
        # 8 = ReLU with alternating
        dict(epochs=200, layer_sizes=[140, 100, 60, 20], activation=nn.ReLU, use_main_effect_nets=True,
             num_samples=30000, valid_size=10000, test_size=10000, std_scale=True,
             my_data_norm=False, lr=1e-3, l1_const=5e-5, dropout_p=0, early_stopping=True, patience=10,
             abs_val=True, verbose=False, order=5, o=2, greedy_heuristic=True, gelu_alt_layer=True),
        # 9 = ReLU with main effects gelu
        dict(epochs=200, layer_sizes=[140, 100, 60, 20], activation=nn.ReLU, use_main_effect_nets=True,
             num_samples=30000, valid_size=10000, test_size=10000, std_scale=True,
             my_data_norm=False, lr=1e-3, l1_const=5e-5, dropout_p=0, early_stopping=True, patience=10,
             abs_val=True, verbose=False, order=5, o=2, greedy_heuristic=True, gelu_main_effects=True),
        # 10 = lr 5e4
        dict(epochs=200, layer_sizes=[140, 100, 60, 20], activation=nn.GELU, use_main_effect_nets=True,
             num_samples=30000, valid_size=10000, test_size=10000, std_scale=True,
             my_data_norm=False, lr=5e-4, l1_const=5e-5, dropout_p=0, early_stopping=True, patience=10,
             abs_val=True, verbose=False, order=5, o=2, greedy_heuristic=True),
        # 11 = no main effects and no std scale
        dict(epochs=200, layer_sizes=[140, 100, 60, 20], activation=nn.GELU, use_main_effect_nets=False,
             num_samples=30000, valid_size=10000, test_size=10000, std_scale=False,
             my_data_norm=False, lr=1e-3, l1_const=5e-5, dropout_p=0, early_stopping=True, patience=10,
             abs_val=True, verbose=False, order=5, o=2, greedy_heuristic=True),
        # 12 = no main effects and gelu alt
        dict(epochs=200, layer_sizes=[140, 100, 60, 20], activation=nn.ReLU, use_main_effect_nets=False,
             num_samples=30000, valid_size=10000, test_size=10000, std_scale=True,
             my_data_norm=False, lr=1e-3, l1_const=5e-5, dropout_p=0, early_stopping=True, patience=10,
             abs_val=True, verbose=False, order=5, o=2, greedy_heuristic=True, gelu_alt_layer=True),
        # 13 = no main effects and more layer 180
        dict(epochs=200, layer_sizes=[180, 140, 100, 60, 20], activation=nn.GELU, use_main_effect_nets=False,
             num_samples=30000, valid_size=10000, test_size=10000, std_scale=True,
             my_data_norm=False, lr=1e-3, l1_const=5e-5, dropout_p=0, early_stopping=True, patience=10,
             abs_val=True, verbose=False, order=5, o=2, greedy_heuristic=True),
        # 14 = no main effects + no l1_const -- this is best
        dict(epochs=200, layer_sizes=[140, 100, 60, 20], activation=nn.GELU, use_main_effect_nets=False,
             num_samples=30000, valid_size=10000, test_size=10000, std_scale=True,
             my_data_norm=False, lr=1e-3, l1_const=0, dropout_p=0, early_stopping=True, patience=10,
             abs_val=True, verbose=False, order=5, o=2, greedy_heuristic=True),
        # 15 = no l1_const
        dict(epochs=200, layer_sizes=[140, 100, 60, 20], activation=nn.GELU, use_main_effect_nets=True,
             num_samples=30000, valid_size=10000, test_size=10000, std_scale=True,
             my_data_norm=False, lr=1e-3, l1_const=0, dropout_p=0, early_stopping=True, patience=10,
             abs_val=True, verbose=False, order=5, o=2, greedy_heuristic=True),
    ]
    params = params[args.use_default_param_set - 1]

# Run experiments  -  testing 3, 14,  and 1
input_size = 10
results = [[] for _ in range(input_size)]
means = []
stds = []
for i in range(10):
    for j in range(args.num_trials):
        print("Experiment ", i + 1)
        print("Trial ", j + 1)
        print("Params: ", params)

        mine, nid = run(**params, num=i + 1)

        print("My AUC ", mine, " NID AUC", nid)
        results[i].append((mine, nid))
    means.append((np.mean(np.array([pair[0] for pair in results[i]]), 0), np.mean(np.array([pair[1] for pair in results[i]]), 0)))
    stds.append((np.std(np.array([pair[0] for pair in results[i]]), 0), np.std(np.array([pair[1] for pair in results[i]]), 0)))

try:
    os.makedirs("./Results/{}/".format(args.use_default_param_set))
except:
    pass
file_object = open('./Results/{}/means.txt'.format(args.use_default_param_set), 'w')
file_object.write(str(means))
file_object.close()
file_object = open('./Results/{}/stds.txt'.format(args.use_default_param_set), 'w')
file_object.write(str(stds))
file_object.close()

# Print results
print("Results")
print(np.array(results))
print("Experiments Averaged (Mine)")
print("Mean:")
print(np.mean(np.array([r[0] for r in means]), 0))
print("Standard Deviation:")
print(np.mean(np.array([r[0] for r in stds]), 0))
print("Experiments Averaged (NID)")
print("Mean:")
print(np.mean(np.array([r[1] for r in means]), 0))
print("Standard Deviation:")
print(np.mean(np.array([r[1] for r in stds]), 0))
print("Params")
print(params, dict(num_trials=args.num_trials))

# Print latex tables
rows = ["$F_{" + str(experiment) + "}(\mathbf{x})$" for experiment in range(1, 11)]
if params["order"] == params["o"]:
    cols = [
        ["MedMeanClose", "MMMMMClose", "MedMinModeClose"],
        ["MMMMMCloseMed", "MMMMMCloseMax"], ["MMMMMCloseMin", "MMMMMCloseMode"],
    ]
    all_means = np.around([list([e[0] for e in experiment[0]]) + [experiment[1][0][0]] for experiment in means], decimals=3)
    all_stds = np.around([list([e[0] for e in experiment[0]]) + [experiment[1][0][0]] for experiment in stds], decimals=3)
    line_marker = cols[-1][-1]
    cols[-1].append("NID (State of Art)")
else:
    cols = [["".join(s)] for s in list(powerset(["Mean", "Med", "Min", "Max", "Mode", "Rand"]))]
    cols = [[mode_name + col[0]] for col in cols for mode_name in ["MeanOf", "MedOf", "MinOf", "MaxOf", "ModeOf"]]
    cols = [[col + " {}-Way".format(N) for N in range(2, params["order"] + 1) for pair in zip(sub_cols, ["NID" for _ in range(len(sub_cols))]) for col in pair] for sub_cols in cols]
    cols = [col[i:i+2] for col in cols for i in range(0, len(col), 2)]
    all_means = np.around([[item for pair in zip(list(experiment[0]), list(experiment[1])) for item in np.ravel([pair[0], pair[1]], "F")] for experiment in means], decimals=3)
    all_stds = np.around([[item for pair in zip(list(experiment[0]), list(experiment[1])) for item in np.ravel([pair[0], pair[1]], "F")] for experiment in stds], decimals=3)
    line_marker = "None"
l = print_latex_tables(rows, cols, all_means, all_stds, line_marker=line_marker)

experiment_means = np.squeeze(np.around(np.mean(all_means, 0), decimals=3).reshape((-1, 2))[:, 0])
top_2_way = sorted(zip([col[0] for col in cols], experiment_means), key=lambda x: x[1] if "2-Way" in x[0] else -np.inf, reverse=True)
top_3_way = sorted(zip([col[0] for col in cols], experiment_means), key=lambda x: x[1] if "3-Way" in x[0] else -np.inf, reverse=True)
top_4_way = sorted(zip([col[0] for col in cols], experiment_means), key=lambda x: x[1] if "4-Way" in x[0] else -np.inf, reverse=True)
top_5_way = sorted(zip([col[0] for col in cols], experiment_means), key=lambda x: x[1] if "5-Way" in x[0] else -np.inf, reverse=True)
avg_way = np.squeeze(np.reshape(experiment_means, newshape=[-1, 4]).mean(1))
top_avg_way = sorted(zip([col[0].replace("2-Way", "") for col in cols[0::4]], avg_way), key=lambda x: x[1], reverse=True)
print("My Results:")
print("Top 2-Way: ", top_2_way)
print("Top 3-Way: ", top_3_way)
print("Top 4-Way: ", top_4_way)
print("Top 5-Way: ", top_5_way)
print("Top Average-Way: ", top_avg_way)
nid_experiment_means = np.squeeze(np.around(np.mean(all_means, 0), decimals=3).reshape((-1, 2))[:, 1])
nid_top_2_way = sorted(zip([col[0] for col in cols], nid_experiment_means), key=lambda x: x[1] if "2-Way" in x[0] else -np.inf, reverse=True)
nid_top_3_way = sorted(zip([col[0] for col in cols], nid_experiment_means), key=lambda x: x[1] if "3-Way" in x[0] else -np.inf, reverse=True)
nid_top_4_way = sorted(zip([col[0] for col in cols], nid_experiment_means), key=lambda x: x[1] if "4-Way" in x[0] else -np.inf, reverse=True)
nid_top_5_way = sorted(zip([col[0] for col in cols], nid_experiment_means), key=lambda x: x[1] if "5-Way" in x[0] else -np.inf, reverse=True)
nid_avg_way = np.squeeze(np.reshape(nid_experiment_means, (-1, 4)).mean(1))
nid_top_avg_way = sorted(zip([col[0].replace("2-Way", "") for col in cols[0::4]], nid_avg_way), key=lambda x: x[1], reverse=True)
print("NID Results:")
print("Top 2-Way: ", nid_top_2_way)
print("Top 3-Way: ", nid_top_3_way)
print("Top 4-Way: ", nid_top_4_way)
print("Top 5-Way: ", nid_top_5_way)
print("Top Average-Way: ", nid_top_avg_way)

file_object = open('./Results/{}/output.txt'.format(args.use_default_param_set), 'w')
file_object.write(l)
file_object.write("\n\n\nMy Results:")
file_object.write("\nTop 2-Way: {}".format(top_2_way))
file_object.write("\nTop 3-Way: {}".format(top_3_way))
file_object.write("\nTop 4-Way: {}".format(top_4_way))
file_object.write("\nTop 5-Way: {}".format(top_5_way))
file_object.write("\nTop Average-Way: {}".format(top_avg_way))
file_object.write("\nNID Results:")
file_object.write("\nTop 2-Way: {}".format(nid_top_2_way))
file_object.write("\nTop 3-Way: {}".format(nid_top_3_way))
file_object.write("\nTop 4-Way: {}".format(nid_top_4_way))
file_object.write("\nTop 5-Way: {}".format(nid_top_5_way))
file_object.write("\nTop Average-Way: {}".format(nid_top_avg_way))
file_object.close()


