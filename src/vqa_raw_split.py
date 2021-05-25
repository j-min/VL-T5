import json
from tqdm import tqdm
if __name__ == '__main__':

    vqa_data_dir = '/ssd-playpen/home/jmincho/workspace/datasets/vqa'
    import os
    os.chdirs(vqa_data_dir)

    with open('./v2_mscoco_train2014_annotations.json') as f:
        train2014_data = json.load(f)

    with open('./v2_mscoco_val2014_annotations.json') as f:
        val2014_data = json.load(f)

    train2014_id2datum = {}
    for datum in train2014_data['annotations']:
        qid = datum['question_id']
        train2014_id2datum[qid] = datum

    val2014_id2datum = {}
    for datum in val2014_data['annotations']:
        qid = datum['question_id']
        val2014_id2datum[qid] = datum

    # for split in tqdm(['train', 'minival', 'nominival']):
    #     with open(f'./{split}.json') as f:
    #         lxmert_data = json.load(f)

    #     data = lxmert_data
    #     raw_data = []
    #     qids = []
    #     for datum in data:
    #         img_id = datum['img_id']
    #         qid = datum['question_id']
    #         qids.append(qid)
    #         if 'val' in img_id:
    #             raw_datum = val2014_id2datum[qid]
    #         elif 'train' in img_id:
    #             raw_datum = train2014_id2datum[qid]
    #         else:
    #             print(f'something wrong: img id {img_id} / qid {qid}')
    #         raw_data.append(raw_datum)

    #     with open(f'./{split}_raw.json', 'w') as f:
    #         json.dump(raw_data, f)
    #     with open(f'./{split}_ids.json', 'w') as f:
    #         json.dump(qids, f)

    for split in tqdm(['train', 'minival', 'nominival']):
        with open(f'./{split}.json') as f:
            lxmert_data = json.load(f)

        data = lxmert_data
        raw_data = []
        qids = []
        for datum in data:
            img_id = datum['img_id']
            qid = datum['question_id']
            qids.append(qid)
            if 'val' in img_id:
                raw_datum = val2014_id2datum[qid]
            elif 'train' in img_id:
                raw_datum = train2014_id2datum[qid]
            else:
                print(f'something wrong: img id {img_id} / qid {qid}')
            datum.update(raw_datum)

    #         datum['answers'] = raw_datum['answers']
    #         datum['answer_type'] = raw_datum['other']

        with open(f'./{split}_raw.json', 'w') as f:
            json.dump(data, f)
        with open(f'./{split}_ids.json', 'w') as f:
            json.dump(qids, f)

    total_data = []
    for split in ['train', 'nominival', 'minival']:
        with open(f'./{split}.json') as f:
            lxmert_data = json.load(f)
        total_data.extend(lxmert_data)

    for datum in total_data:
        if 'val' in datum['img_id']:
            datum['answers'] = val2014_id2datum[datum['question_id']]['answers']
        elif 'train' in datum['img_id']:
            datum['answers'] = train2014_id2datum[datum['question_id']]['answers']

    print('# total questions', len(total_data))
    unanswerable_data = list(filter(lambda x: len(x['label']) == 0, total_data))
    answerable_data = list(filter(lambda x: len(x['label']) > 0, total_data))
    print('# unanswerable questions', len(unanswerable_data))
    print('# answerable questions', len(answerable_data))

    with open('./answerable.json', 'w') as f:
        json.dump(answerable_data, f)

    with open('./unanswerable.json', 'w') as f:
        json.dump(unanswerable_data, f)
