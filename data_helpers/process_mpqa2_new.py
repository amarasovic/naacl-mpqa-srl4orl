from nltk.tokenize import word_tokenize
from nltk.tokenize import WhitespaceTokenizer
import codecs
import numpy as np
import logging
from collections import defaultdict
import string
#from colorama import init
#from colorama import Fore, Back, Style
import json

import _ctypes
import json
import re


global valid_sent
global valid_sent_offsets
global begin_sequence
global end_sequence


def collect_opinion_entities(argv):
    agents = defaultdict(list)
    agents_uncertain = defaultdict(list)
    attitudes = defaultdict(list)
    attitudes_type = defaultdict(list)
    attitudes_inferred = defaultdict(list)
    attitudes_sarcastic = defaultdict(list)
    attitudes_repetition = defaultdict(list)
    attitudes_contrast = defaultdict(list)
    attitudes_uncertain = defaultdict(list)
    targets = defaultdict(list)
    targets_uncertain = defaultdict(list)

    filename = argv[0]
    file_lre = codecs.open(filename, "r", encoding="utf-8").readlines()
    # construct an agents dictionary: agents[agent] = (start_agent, end_agent)
    for (i, line) in enumerate(file_lre):
        line = line.strip()
        line_tab = line.split("\t")
        if i < 5:
            continue

        if line_tab[3] == "GATE_agent" and len(line_tab) >= 5:
            try:
                agent = line_tab[4].split("nested-source=")[1].split("\n")[0].split('"')[1].replace(', ', ',')
            except IndexError:
                # 1	4578,4581	string	GATE_agent	 agent-uncertain="somewhat-uncertain"
                try:
                    agent = line_tab[4].split("id=")[1].split("\n")[0].split('"')[1].replace(', ', ',')
                except IndexError:
                    pass

            if len(line_tab[4].split("agent-uncertain=")) > 1:
                agent_uncertain = line_tab[4].split("agent-uncertain=")[1].split('"')[1]
                if not agent_uncertain:
                    agent_uncertain = 'unk'
            else:
                agent_uncertain = 'no'

            if agent:
                agents[agent].append([int(x) for x in line_tab[1].split(',')])
                agents_uncertain[agent] = agent_uncertain

    # construct an attitude dictionaries:
    # 1) attitudes[attitude_id] = [target_ids]
    # 2) attitudes_types[attitude_id] = [attitude_type]
    for i, line in enumerate(file_lre):
        if i < 5:
            continue  # first 5 lines are meta-data

        line_tab = line.split("\t")
        if line_tab[3] == "GATE_attitude" and len(line_tab[4].split(' ')) > 1:
            if len(line_tab[4].split("target-link=")) > 1:
                target_ids = line_tab[4].split("target-link=")[1].split("\n")[0].split('"')[1].replace(' ', '').split(',')
                if not target_ids:
                    target_ids = ['none']
            else:
                target_ids = ['none']

            if len(line_tab[4].split("attitude-type=")) > 1:
                type_full = line_tab[4].split("attitude-type=")[1].split('"')[1]
            else:
                type_full = 'none'

            if len(line_tab[4].split("inferred=")) > 1:
                inferred = line_tab[4].split("inferred=")[1].split('"')[1]
                if not inferred:
                    inferred = 'unk'
            else:
                inferred = 'no'

            if len(line_tab[4].split("sarcastic=")) > 1:
                sarcastic = line_tab[4].split("sarcastic=")[1].split('"')[1]
                if not sarcastic:
                    sarcastic = 'unk'
            else:
                sarcastic = 'no'

            if len(line_tab[4].split("repetition=")) > 1:
                repetition = line_tab[4].split("repetition=")[1].split('"')[1]
                if not repetition:
                    repetition = 'unk'
            else:
                repetition = 'no'

            if len(line_tab[4].split("contrast=")) > 1:
                contrast = line_tab[4].split("contrast=")[1].split('"')[1]
                if not contrast:
                    contrast = 'unk'
            else:
                contrast = 'no'

            if len(line_tab[4].split("attitude-uncertain=")) > 1:
                attitude_uncertain = line_tab[4].split("attitude-uncertain=")[1].split('"')[1]
                if not attitude_uncertain:
                    attitude_uncertain = 'unk'
            else:
                attitude_uncertain = 'no'

            if len(line_tab[4].split("id=")) > 1:
                attitude_id = line_tab[4].split("id=")[1].split("\n")[0].split('"')[1].replace(', ', ',')
                attitudes[attitude_id].extend([target_id for target_id in target_ids])
                attitudes_type[attitude_id] = type_full
                attitudes_inferred[attitude_id] = inferred
                attitudes_sarcastic[attitude_id] = sarcastic
                attitudes_repetition[attitude_id] = repetition
                attitudes_contrast[attitude_id] = contrast
                attitudes_uncertain[attitude_id] = attitude_uncertain

            else:
                # assert that the attitude does not have an id because it is an inferred attitude
                assert len(line_tab[4].split("inferred=")) > 1

    # construct a target dictionary: targets[target_id] = (start_target, end_target)
    for (i, line) in enumerate(file_lre):
        if i < 5:
            continue
        line_tab = line.split("\t")

        if line_tab[3] == "GATE_target" and len(line_tab) > 4 and len(line_tab[4].split("id=")) > 1:
            target = line_tab[4].split("id=")[1].split("\n")[0].split('"')[1].replace(', ', ',')

            if len(line_tab[4].split("target-uncertain=")) > 1:
                target_uncertain = line_tab[4].split("target-uncertain=")[1].split('"')[1]
                if not target_uncertain:
                    target_uncertain = 'unk'
            else:
                target_uncertain = 'no'
            targets[target].append([int(x) for x in line_tab[1].split(',')])
            targets_uncertain[target] = target_uncertain

    return agents, agents_uncertain, attitudes, attitudes_inferred, attitudes_type, attitudes_sarcastic, \
           attitudes_repetition, attitudes_contrast, attitudes_uncertain, targets, targets_uncertain


def label(entity_tuple, entity_tag, ds_dict, att_dict=None, flag=0, sent_ds_id=-1):
    global valid_sent
    global valid_sent_offsets
    global begin_sequence
    global end_sequence

    start_entity = int(entity_tuple[0])
    end_entity = int(entity_tuple[1])

    c1 = np.where(begin_sequence <= start_entity)

    c2 = np.where(end_entity <= end_sequence)

    if len(np.intersect1d(c1, c2)) > 1:
        logging.info("two sentences with this entity")
        flag = 2
        return ds_dict, att_dict, flag, -1

    if len(np.intersect1d(c1, c2)) == 0:
        logging.info("no such sentence")
        flag = 3
        return ds_dict, att_dict, flag, -1

    # index of a sentence which contains the direct-subjective
    sent_id = int(np.intersect1d(c1, c2))

    # target has to be in the same sentence as the direct-subjective
    if entity_tag in ['h', 't'] and sent_id != sent_ds_id:
        return ds_dict, att_dict, flag, sent_id

    sentence_tokenized = valid_sent[sent_id]
    sentence_offsets = valid_sent_offsets[sent_id]

    start_entity = -1
    end_entity = -1
    # find token index with char offsets
    for idx, offset in enumerate(sentence_offsets):
        if offset[0] <= entity_tuple[0] <= offset[1]:
            start_entity = idx
        if offset[0] <= entity_tuple[1] <= offset[1]:
            end_entity = idx

    if start_entity != -1 and end_entity == -1:
        for idx, offset in enumerate(sentence_offsets):
            if offset[1] <= entity_tuple[1]:
                end_entity = idx

    if end_entity != -1 and start_entity == -1:
        for idx, offset in enumerate(sentence_offsets):
            if offset[0] <= entity_tuple[0]:
                start_entity = idx

    entity_indices = []
    if start_entity != -1 and end_entity != -1:
        entity_indices = range(start_entity, end_entity+1)
        entity_tokenize = sentence_tokenized[start_entity:end_entity+1]

    if entity_indices:
        if entity_tokenize[-1] in string.punctuation:
            entity_tokenize.pop()
            entity_indices.pop()

        if entity_tag == 't':
            att_dict['targets_tokenized'].append(entity_tokenize)
            att_dict['targets_indices'].append(entity_indices)

        if entity_tag == 'h':
            ds_dict['holders_tokenized'].append(entity_tokenize)
            ds_dict['holders_indices'].append(entity_indices)

        if entity_tag == 'd':
            ds_dict['ds_tokenized'] = entity_tokenize
            ds_dict['ds_indices'] = entity_indices
    else:
        # due to sentence splitting and tokenization, it can happen that we can not assign tokens
        if entity_tag == "d":
            flag = 1
    return ds_dict, att_dict, flag, sent_id


def get_annotations(argv):
    global valid_sent
    global valid_sent_offsets
    global begin_sequence
    global end_sequence

    doc_dict = {}
    # lre file - MPQA annotations
    filename = argv[0]
    file_lre = codecs.open(filename, "r", encoding="utf-8").readlines()

    file_doc = []  # strings of valid sentences
    valid_sent = []  # tokenized valid sentences
    valid_sent_offsets = []  # char offsets of tokens of valid sentences
    document = codecs.open(argv[1], "r", encoding="utf-8").read()
    document_corenlp = argv[3]

    # collect tokenized sentences and char offsets
    beginnings = []
    endings = []
    for sent in document_corenlp:
        beginnings.append(sent['char_offsets'][0][0])
        endings.append(sent['char_offsets'][len(sent['char_offsets']) - 1][1] + 1)
        file_doc.append(document[sent['char_offsets'][0][0]:sent['char_offsets'][len(sent['char_offsets']) - 1][1] + 1])
        valid_sent.append(sent['tokens'])
        valid_sent_offsets.append(sent['char_offsets'])

    begin_sequence = np.asarray(beginnings)
    end_sequence = np.asarray(endings)

    # number of valid sentences
    valid_sent_num = len(valid_sent)

    doc_dict.update({'document_path': argv[1], 'sentences_num': valid_sent_num})

    for i, sent in enumerate(valid_sent):
        sentence_name = "sentence" + str(i)
        doc_dict.update({sentence_name: {'sentence_tokenized': sent, 'dss_ids': []}})

    # maximum sentence length (= # of words)
    max_length = 0
    for sent in file_doc:
        max_length = max(max_length, len(sent))

    agents, agents_uncertain, attitudes, attitudes_inferred, attitudes_type, attitudes_sarcastic, \
    attitudes_repetition, attitudes_contrast, attitudes_uncertain, targets, targets_uncertain = collect_opinion_entities(argv)

    # labelling
    ds_id = 0
    implicit_ds = 0
    writer_ds = 0
    one_char_ds = 0
    count_ds = 0
    two_sent_ds = 0
    no_sent_ds_one_char_ds = 0
    two_sent_ds_one_char_ds = 0
    no_sent_ds = 0
    target_not_where_ds = 0
    holder_not_where_ds = 0
    target_not_collected = 0
    holder_not_collected = 0
    one_char_ds_implicit = 0

    all_ds = 0

    for (i, line) in enumerate(file_lre):
        line = line.strip()
        if i < 5:
            continue

        line_tab = line.split("\t")

        if line_tab[3] == "GATE_direct-subjective" and len(line_tab) > 4:
            all_ds += 1
            flag = 0
            start_ds = int(line_tab[1].split(",")[0])
            end_ds = int(line_tab[1].split(",")[1])

            # discard one character long direct subjectives
            if end_ds - start_ds <= 1:
                one_char_ds += 1
                continue

            if end_ds - start_ds > 1:
                ds_dict = {'ds_tokenized': None,
                           'ds_indices': None,
                           'ds_implicit': False,
                           'ds_polarity': None,
                           'ds_intensity': None,
                           'ds_expression_intensity': None,
                           'ds_annotation_uncertain': None,
                           'ds_subjective_uncertain': None,
                           'ds_insubstantial': None,
                           'attitude_link_exists': True,
                           'attitudes': [],
                           'att_num': 0,
                           'holders_tokenized': [],
                           'holders_indices': [],
                           'holders_uncertain': [],
                           'holder_ds_overlap': []
                           }

                ds_dict, att_dict, flag, sent_ds_id = label((start_ds, end_ds), 'd', ds_dict, att_dict=None, flag=0, sent_ds_id=-1)

                # we need the index of the sentences with the ds to request that opinion roles are in the same sentence
                assert sent_ds_id is not None

            if flag == 1:
                print line
                continue

            if flag == 2:
                two_sent_ds += 1
                continue

            if flag == 3:
                no_sent_ds += 1
                continue

            if flag == 0:
                count_ds += 1
                arguments = line_tab[4]

                if len(arguments.split("polarity=")) > 1:
                    ds_polarity = arguments.split("polarity=")[1].split("\n")[0].split('"')[1].replace(', ', ',')
                    if not ds_polarity:
                        ds_polarity = 'unk'
                    ds_dict['ds_polarity'] = ds_polarity
                else:
                    ds_dict['ds_polarity'] = "none"

                if len(arguments.split("intensity=")) > 1:
                    ds_intensity = arguments.split("intensity=")[1].split("\n")[0].split('"')[1].replace(', ', ',')
                    if not ds_intensity:
                        ds_intensity = 'unk'
                    ds_dict['ds_intensity'] = ds_intensity
                else:
                    ds_dict['ds_intensity'] = "none"

                if len(arguments.split("expression-intensity=")) > 1:
                    expression_intensity = arguments.split("expression-intensity=")[1].split("\n")[0].split('"')[1].replace(', ', ',')
                    if not expression_intensity:
                        expression_intensity = 'unk'
                    ds_dict['ds_expression_intensity'] = expression_intensity
                else:
                    ds_dict['ds_expression_intensity'] = "none"

                if len(arguments.split("annotation-uncertain=")) > 1:
                    annotation_uncertain = arguments.split("annotation-uncertain=")[1].split("\n")[0].split('"')[1].replace(', ', ',')
                    if not annotation_uncertain:
                        annotation_uncertain = 'unk'
                    ds_dict['ds_annotation_uncertain'] = annotation_uncertain
                else:
                    ds_dict['ds_annotation_uncertain'] = "none"

                if len(arguments.split("subjective-uncertain=")) > 1:
                    subjective_uncertain = arguments.split("subjective-uncertain=")[1].split("\n")[0].split('"')[1].replace(', ', ',')
                    if not subjective_uncertain:
                        subjective_uncertain = 'unk'
                    ds_dict['ds_subjective_uncertain'] = subjective_uncertain
                else:
                    ds_dict['ds_subjective_uncertain'] = "none"

                if len(arguments.split("insubstantial=")) > 1:
                    insubstantial = arguments.split("insubstantial=")[1].split("\n")[0].split('"')[1].replace(', ', ',')
                    if not insubstantial:
                        insubstantial = 'unk'
                    ds_dict['ds_insubstantial'] = insubstantial
                else:
                    ds_dict['ds_insubstantial'] = "none"

                # attitude-link attribute can be missing
                # 2053	3765,3769	string	GATE_direct-subjective	 nested-source="w,mccoy" intensity=""
                # in these cases the target can't be traced
                if len(arguments.split("attitude-link=")) <= 1:
                    ds_dict['attitude_link_exists'] = False

                else:
                    # attitude-link="a4, a6" ---> a4,a6
                    attitude_ids = arguments.split("attitude-link=")[1].split('"')[1].replace(' ', '').split(',')
                    ds_dict['att_num'] = len(attitude_ids)

                    for aid, att_id in enumerate(attitude_ids):
                        # get list of target ids, e.g. ['t20'], ['t20', 't10']
                        attitude_dict = {
                                         'targets_tokenized': [],
                                         'targets_indices': [],
                                         'targets_uncertain': [],
                                         'target_ds_overlap': []
                                         }
                        att_type = attitudes_type[att_id] if attitudes_type[att_id] else 'none'
                        attitude_dict['attitudes_types'] = att_type
                        ds_dict['attitudes'].append(att_type)
                        attitude_dict['attitudes_inferred'] = attitudes_inferred[att_id] if attitudes_inferred[att_id] else 'none'
                        attitude_dict['attitudes_sarcastic'] = attitudes_sarcastic[att_id] if attitudes_sarcastic[att_id] else 'none'
                        attitude_dict['attitudes_repetition'] = attitudes_repetition[att_id] if attitudes_repetition[att_id] else 'none'
                        attitude_dict['attitudes_contrast'] = attitudes_contrast[att_id] if attitudes_contrast[att_id] else 'none'
                        attitude_dict['attitudes_uncertain'] = attitudes_uncertain[att_id] if attitudes_uncertain[att_id] else 'none'

                        target_id_strings = attitudes[att_id]

                        for t, target_id_string in enumerate(target_id_strings):
                            if target_id_string != 'none':
                                target_ranges_list = targets[target_id_string]
                                if not target_ranges_list:
                                    target_not_collected += 1

                                if len(target_ranges_list) == 1: # if the target appear only once in the document
                                    target_range = target_ranges_list[0]
                                    if len(set(range(start_ds, end_ds)) & set(range(target_range[0], target_range[1]))) > 1:
                                        ds_dict, attitude_dict, flag, sent_t_id = label(target_range, "t", ds_dict,
                                                                                        attitude_dict, flag, sent_ds_id)
                                        if sent_t_id != sent_ds_id:
                                            target_not_where_ds += 1
                                        else:
                                            attitude_dict['target_ds_overlap'].append(True)
                                            attitude_dict['targets_uncertain'].append(targets_uncertain[target_id_string])
                                    else:
                                        ds_dict, attitude_dict, flag, sent_t_id = label(target_range, "t", ds_dict, attitude_dict, flag, sent_ds_id)
                                        if sent_t_id != sent_ds_id:
                                            target_not_where_ds += 1
                                        else:
                                            attitude_dict['target_ds_overlap'].append(False)
                                            attitude_dict['targets_uncertain'].append(targets_uncertain[target_id_string])

                                if len(target_ranges_list) > 1:

                                    # if the target appear more than once in the document (weird)
                                    # choose the closest
                                    overlapping_ranges = []
                                    for j, target_range in enumerate(target_ranges_list):
                                        if len(set(range(start_ds, end_ds)) & set(range(target_range[0], target_range[1]))):
                                            overlapping_ranges.append(j)
                                    target_ranges_clean = [agent_range for k, agent_range in enumerate(target_ranges_list) if k not in overlapping_ranges]

                                    # if there is no target that does not overlap with ds then pick the closest
                                    overlap = False
                                    if not target_ranges_clean:
                                        overlap = True
                                        target_ranges_clean = target_ranges_list

                                    min_dist = 1000
                                    index_min_dist = -1
                                    for j, target_range in enumerate(target_ranges_list):
                                        start_target = target_range[0]
                                        end_target = target_range[1]

                                        if end_target <= start_ds:
                                            dist = start_ds - end_target

                                        if end_ds <= start_target:
                                            dist = start_target - end_ds

                                        if (start_target <= start_ds) and (end_target <= end_ds):
                                            dist = min(start_ds - start_target, end_ds - end_target)

                                        if (start_ds <= start_target) and (end_ds <= end_target):
                                            dist = min(start_target - start_ds, end_target - end_ds)

                                        if (start_ds <= start_target) and (end_target <= end_ds):
                                            dist = min(start_ds - start_target, end_ds - end_target)

                                        if (start_target <= start_ds) and (end_ds <= end_target):
                                            dist = min(start_ds - start_target, end_target - end_ds)

                                        if dist < min_dist:
                                            min_dist = dist
                                            index_min_dist = j

                                    target_range = target_ranges_clean[index_min_dist]
                                    ds_dict, attitude_dict, flag, sent_t_id = label(target_range, "t", ds_dict,
                                                                                    attitude_dict, flag, sent_ds_id)
                                    if sent_t_id != sent_ds_id:
                                        target_not_where_ds += 1
                                    else:
                                        attitude_dict['target_ds_overlap'].append(overlap)
                                        attitude_dict['targets_uncertain'].append(targets_uncertain[target_id_string])
                        ds_dict['att'+str(aid)] = attitude_dict

                if len(line_tab[4].split("implicit=")) > 1:
                    ds_dict['ds_implicit'] = True
                    implicit_ds += 1

                if not ds_dict['ds_implicit'] and len(line_tab[4].split("nested-source=")) > 1:
                    # discard direct subjectives expressed by writer only (these should be marked as implicit)
                    if line_tab[4].split("nested-source=")[1].split("\n")[0].split('"')[1].replace(' ', '') in ['w', 'w,implicit'] or \
                                    line_tab[4].split("nested-source=")[1].split("\n")[0].split('"')[1].replace(' ', '').split(',')[-1] in ['w', 'implicit']:
                        ds_dict['ds_implicit'] = True
                        implicit_ds += 1

                if len(arguments.split("nested-source=")) > 1 and not ds_dict['ds_implicit']:
                    #agent = arguments.split("nested-agent=")[1].split('" ')[0]
                    agent = line_tab[4].split("nested-source=")[1].split("\n")[0].split('"')[1].replace(', ', ',')

                    # try to fix agent
                    # nhs -> w,nhs
                    # w,ip -> ip
                    # w,mug,mug -> w,mug
                    if agent not in agents:
                        flag = False
                        if len(agent.split('w,')) <= 1:
                            agent_new = 'w,' + agent
                            if agent_new in agents:
                                agent = agent_new
                                flag = True
                        if not flag and len(agent.split('w,')) > 1:
                            agent_new = agent.split('w,')[1]
                            if agent_new in agents:
                                agent = agent_new
                                flag = True
                        if not flag:
                            agent_split = agent.split(',')
                            marked = []
                            agent_new = ''
                            for a in agent_split:
                                if a not in marked:
                                    agent_new += a + ','
                                    marked.append(a)

                            agent_final = ','.join(agent_new.split(',')[:-1])

                            if agent_final in agents:
                                agent = agent_final
                                flag = True

                    if agent in agents:
                        agent_ranges = agents[agent]
                        if len(agent_ranges) == 1:
                            agent_range = agent_ranges[0]
                            if agent_range != [0, 0]:
                                if len(set(range(start_ds, end_ds)) & set(range(agent_range[0], agent_range[1]))) > 1:
                                    att_dict = None
                                    ds_dict, _, flag, sent_h_id = label(agent_range, "h", ds_dict, att_dict, flag, sent_ds_id)
                                    if sent_h_id != sent_ds_id:
                                        holder_not_where_ds += 1
                                    else:
                                        ds_dict['holder_ds_overlap'].append(True)
                                        ds_dict['holders_uncertain'].append(agents_uncertain[agent])
                                else:
                                    att_dict = None
                                    ds_dict, _, flag, sent_h_id = label(agent_range, "h", ds_dict, att_dict, flag, sent_ds_id)
                                    if sent_h_id != sent_ds_id:
                                        holder_not_where_ds += 1
                                    else:
                                        ds_dict['holder_ds_overlap'].append(False)
                                        ds_dict['holders_uncertain'].append(agents_uncertain[agent])
                        else:
                            overlapping_ranges = []
                            for j, agent_range in enumerate(agent_ranges):
                                if len(set(range(start_ds, end_ds)) & set(range(agent_range[0], agent_range[1]))):
                                    overlapping_ranges.append(j)
                            agent_ranges_clean = [agent_range for k, agent_range in enumerate(agent_ranges) if k not in overlapping_ranges]

                            overlap = False
                            if not agent_ranges_clean:
                                overlap = True
                                agent_ranges_clean = agent_ranges

                            # nested-agent argument can point to agent that occurs more times in the document
                            # pick the closest of those
                            min_dist = 1000
                            index_min_dist = -1
                            for j, agent_range in enumerate(agent_ranges_clean):
                                start_agent = agent_range[0]
                                end_agent = agent_range[1]

                                if end_agent <= start_ds:
                                    dist = start_ds - end_agent

                                if end_ds <= start_agent:
                                    dist = start_agent - end_ds

                                if (start_agent <= start_ds) and (end_agent <= end_ds):
                                    dist = min(start_ds - start_agent, end_ds - end_agent)

                                if (start_ds <= start_agent) and (end_ds <= end_agent):
                                    dist = min(start_agent - start_ds, end_agent - end_ds)

                                if (start_ds <= start_agent) and (end_agent <= end_ds):
                                    dist = min(start_ds-start_agent, end_ds - end_agent)

                                if (start_agent <= start_ds) and (end_ds <= end_agent):
                                    dist = min(start_ds - start_agent, end_agent - end_ds)

                                if dist < min_dist:
                                    min_dist = dist
                                    index_min_dist = j

                            agent_range = agent_ranges_clean[index_min_dist]
                            if agent_range is not [0, 0]:
                                att_dict = None
                                ds_dict, _, flag, sent_h_id = label(agent_range, "h", ds_dict, att_dict, flag, sent_ds_id)
                                if sent_h_id != sent_ds_id:
                                    holder_not_where_ds += 1
                                else:
                                    ds_dict['holder_ds_overlap'].append(overlap)
                                    ds_dict['holders_uncertain'].append(agents_uncertain[agent])
                    else:
                        holder_not_collected += 1

                ds_name = "ds" + str(ds_id)
                sent_name = "sentence"+str(sent_ds_id)

                doc_dict[sent_name][ds_name] = ds_dict

                ds_ids = doc_dict[sent_name]['dss_ids']
                ds_ids.append(ds_id)
                doc_dict[sent_name]['dss_ids'] = ds_ids
            ds_id += 1

    stats = [implicit_ds,
             writer_ds,
             one_char_ds,
             count_ds,
             two_sent_ds,
             no_sent_ds,
             target_not_where_ds,
             holder_not_where_ds,
             target_not_collected,
             holder_not_collected,
             no_sent_ds_one_char_ds,
             two_sent_ds_one_char_ds,
             one_char_ds_implicit,
             all_ds]

    return doc_dict, stats

