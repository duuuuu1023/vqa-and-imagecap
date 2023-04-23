import json
if __name__ =='__main__':
    with open('captions_question_train201410000.json','r') as f:
        data_a = json.load(f)
    output_data=[]
    #record_num = 0
    for num in range(len(4,data_a)):
        que = data_a[num]['sent']
        cap = data_a[num]['caption']
        print('问题是：')
        print(que)
        print('标题：')
        print(cap)
        cap_index = input('请输入序号\输入b保存(1是正确，0是错误):                    ')
        try :
            if cap_index=='b':
                length = len(output_data)
                with open('/root/autodl-tmp/catr-master/annotations/captions_question_train2014{}.json'.format(length),'w') as obj:
                    txt = json.dumps(output_data)
                    obj.write(txt)
                obj.close()
                print('保存成功，现在已经成功标注{}条数据！'.format(str(len(output_data))))
                continue
            data_a[num]['cls']=cap_index
            
        except IndexError:
            print('注意范围')
            cap_index = input('请输入序号\输入b保存(1是正确，0是错误):                    ')
            if cap_index=='b':
                length = len(output_data)
                with open('/root/autodl-tmp/catr-master/annotations/captions_question_train2014{}.json'.format(length),'w') as obj:
                    txt = json.dumps(output_data)
                    obj.write(txt)
                obj.close()
                print('保存成功，现在已经成功标注{}条数据！'.format(str(len(output_data))))
                continue
            data_a[num]['cls']=cap_index
        #record_num += 1
        output_data.append(data_a[num])
        record_num = str(len(output_data))
        print('现在已经成功标注{}条数据！'.format(record_num))
        if len(output_data)==1000:
            break
    length = len(output_data)
    with open('/root/autodl-tmp/catr-master/annotations/new_captions_question_train2014{}.json'.format(length),'w') as obj:
        txt = json.dumps(output_data)
        obj.write(txt)
    obj.close()