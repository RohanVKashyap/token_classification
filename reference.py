def collate_fn(data):
    t1 = time.time()
    ret = merge_with(identity, *data)
    #print(type(ret))                    #dictionary
    #print(ret.keys())                   #['input_text', 'label']
    #print(type(ret['input_text']))      #list
    #print(type(ret['label']))           #list
    #print(ret['label'])
    #print(ret['input_text'])
    #print(embed(tuple(ret['input_text'])))
    #print('first one:',len(ret['input_text']))
    #new_input=[]
    #print('I am here:',embed(tuple(['I am here','I am going to mysore'])).shape)
    #for j in ret['input_text']:
       #print('first j:',j)
       #for i in j.split():
           #print('first i:',i)
           #print(embed(i).shape)
           #print('type 1:',i)
           #print('type 2:',type(embed(tuple(i))))
           #print(tuple(i))
           #print('type 3:',embed(i).shape)
           #print('shape:',torch.tensor(embed(tuple(i))).shape)
           #new_input.append(embed(j.split()))

    new_input=[]
    for j in ret['input_text']:
        new_input.append(embed(tuple(j.split())))
    print('new_input',len(new_input))
    print('new_input1',new_input[1].shape)

    ret['input_embeddings']=torch.stack(new_input).view(-1,384)
    print('first embeddings:',ret['input_embeddings'].shape)
    #print('a1 shape:',a1.shape)       
    #print([embed(tuple(tokens)).shape] for tokens in ret['input_text'][0].split())
    #print(embed(tuple(ret['input_text'])).shape)
    #ret['input_embeddings'] = pipe(ret, get('input_text'), tuple, embed)
    #print(ret['input_embeddings'].shape)
    ret['label'] = torch.stack(ret['label']).squeeze(dim=1).to(DEVICE)
    ret['label'] = ret['label'].view(-1,1)
    #ret['label'] = torch.stack(ret['label']).to(DEVICE)
    #print('Hi')
    #print(ret['label'])
    print(ret['input_embeddings'].shape, ret['label'].shape)
    t2 = time.time()
    return ret
