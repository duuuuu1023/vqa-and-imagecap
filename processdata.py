import json
if __name__ =='__main__':
    with open('annotations/train.json','r') as f:
        data_a = json.load(f)
    with open('annotations/captions_train2014.json','r')as f:
        data_c = json.load(f)
    output_data=[]
    #record_num = 0
    for num in range(len(data_a)):
        aa = data_a[num]['img_id']
        cap =[]
        for da in data_c['annotations']:
            if str(da['image_id']) == aa[-6:] and len(str(da['image_id']))==6:
                cap.append(da['caption'])
        if len(cap)==0:
            continue
        print(cap[0])
        print('caption:')
        for index in range(len(cap)):
            #print(index)
            print(str(index)+':  '+cap[index])
        print('question: ')
        print(data_a[num]['sent'])
        print('answer')
        print(data_a[num]['label'])
        cap_index = input('请输入序号\输入b保存')
        try :
            if cap_index=='b':
                length = len(output_data)
                with open('/root/autodl-tmp/catr-master/annotations/captions_question_train2014{}.json'.format(length),'w') as obj:
                    txt = json.dumps(output_data)
                    obj.write(txt)
                obj.close()
                print('保存成功，现在已经成功标注{}条数据！'.format(str(len(output_data))))
                continue
            data_a[num]['caption']=cap[int(cap_index)]
            
        except IndexError:
            print('注意范围')
            cap_index = input('请输入序号\输入b保存')
            if cap_index=='b':
                length = len(output_data)
                with open('/root/autodl-tmp/catr-master/annotations/captions_question_train2014{}.json'.format(length),'w') as obj:
                    txt = json.dumps(output_data)
                    obj.write(txt)
                obj.close()
                print('保存成功，现在已经成功标注{}条数据！'.format(str(len(output_data))))
                continue
            data_a[num]['caption']=cap[int(cap_index)]
        #record_num += 1
        output_data.append(data_a[num])
        record_num = str(len(output_data))
        print('现在已经成功标注{}条数据！'.format(record_num))
        if len(output_data)==1000:
            break
    length = len(output_data)
    with open('/root/autodl-tmp/catr-master/annotations/captions_question_train2014{}.json'.format(length),'w') as obj:
        txt = json.dumps(output_data)
        obj.write(txt)
    obj.close()