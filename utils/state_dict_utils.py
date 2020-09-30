from collections import OrderedDict
import torch
import numpy as np
import copy

def load_state_dict_partial(model, state_dict):

    model_state_dict = model.state_dict()

    model_state_dict = {k:v for (k,v) in state_dict.items() if k in model_state_dict.keys() }

    model.load_state_dict(model_state_dict)

    return model



def toggle_state_dict(state_dict):

    new_state_dict = {}
    already_added = []

    for ik, k  in enumerate(state_dict.keys()):

        item = state_dict[k]

        if ('bn' not in k) and ('weight' in k) and ('fc' not in k):

            if k not in already_added:
                

                if 'feedback' not in k:
                    item_dual = state_dict[k+'_feedback']
                    if 'upsample' in k:
                        k = k.replace('up','down')
                    elif 'downsample' in k:
                        k = k.replace('down', 'up')

                    new_state_dict.update({k+'_feedback':item})
                    new_state_dict.update({k:item_dual})
                

                else:
                    item_dual = state_dict[k.split('_')[0]]
                    if 'upsample' in k:
                        k = k.replace('up','down')
                    elif 'downsample' in k:
                        k = k.replace('down', 'up')
                    new_state_dict.update({k.split('_')[0]:item})
                    new_state_dict.update({k:item_dual})

        elif ('fc'in k) and ('weight' in k):
            if 'feedback' in k:
                new_state_dict.update({k.strip('_feedback'): torch.transpose(item, 0, 1)})
            else:
                new_state_dict.update({k+'_feedback': torch.transpose(item, 0, 1)})
                
        else:
            new_state_dict.update({k:item})

        already_added.append(k)

    return new_state_dict


def toggle_state_dict_normalize(state_dict):
    # this code copies the forward parameters to the backward parameters and vice versa
    # additionally it normalizes both the forward and backward path 

    new_state_dict = {}
    already_added = []

    for ik, k  in enumerate(state_dict.keys()):

        item = state_dict[k]

        if ('bn' not in k) and ('weight' in k) and ('fc' not in k):

            if k not in already_added:
                

                if 'feedback' not in k:
                    item_dual = state_dict[k+'_feedback']
                    if 'upsample' in k:
                        k = k.replace('up','down')
                    elif 'downsample' in k:
                        k = k.replace('down', 'up')

                    new_state_dict.update({k+'_feedback':item})
                    new_state_dict.update({k:item_dual})

                else:
                    item_dual = state_dict[k.split('_')[0]]
                    if 'upsample' in k:
                        k = k.replace('up','down')
                    elif 'downsample' in k:
                        k = k.replace('down', 'up')
                        
                    denom_item = 1#torch.sqrt(torch.abs(item)**2+torch.abs(item_dual)**2)
                    denom_item_dual = 1#torch.sqrt(torch.abs(item)**2+torch.abs(item_dual)**2)
                    
                    new_state_dict.update({k.split('_')[0]:item/denom_item })
                    new_state_dict.update({k:item_dual/denom_item_dual})

        elif ('fc'in k) and ('weight' in k):
            if 'feedback' in k:
                new_state_dict.update({k.strip('_feedback'): torch.transpose(item, 0, 1)})
            else:
                new_state_dict.update({k+'_feedback': torch.transpose(item, 0, 1)})
                
        else:
            new_state_dict.update({k:item})

        already_added.append(k)

    return new_state_dict

def toggle_state_dict_resnets(state_dict, state_dict_dest):

    """
    Works for modular resnet codes
    """
    # How many layers in resnet?
    layers = [int(k[12]) for k in state_dict.keys() if k.startswith('module.layer')]
    #'check if you wrapped the model in a module like nn.parallel does'
    if len(layers):
        modulestr = 'module.layer'
        blocknumsplit=2
    else:
        modulestr ='layer'
        layers = [int(k[5]) for k in state_dict.keys() if k.startswith('layer')]
        blocknumsplit = 1

    num_layers = max(layers)
    # print(state_dict.keys())
    new_state_dict = {}
    already_added = []

    for ik, k  in enumerate(state_dict.keys()):

        item = state_dict[k]

        if ('bn' not in k) and ('weight' in k) and ('fc' not in k):

            if k not in already_added:
                

                if 'feedback' not in k:
                    item_dual = state_dict[k+'_feedback']
                    if 'upsample' in k:
                        k = k.replace('up','down')
                    elif 'downsample' in k:
                        k = k.replace('down', 'up')

                    new_state_dict.update({k+'_feedback':item})
                    new_state_dict.update({k:item_dual})
                

                else:
                    item_dual = state_dict[k.split('_')[0]]
                    if 'upsample' in k:
                        k = k.replace('up','down')
                    elif 'downsample' in k:
                        k = k.replace('down', 'up')
                        
                    new_state_dict.update({k.split('_')[0]:item})
                    new_state_dict.update({k:item_dual})
        # using BP FC, exlcuding both weight and bias and also  module.biasfc needed for fixup
        elif ('fc'in k): #and ('weight' in k):
            pass
            # new_state_dict.update({k:torch.transpose(item,0,1)})
        
        else:
            new_state_dict.update({k:item})

        already_added.append(k)

    
    # to revert the block number for toggling
    num_blocks_in_layer1 = []
    num_blocks_in_layer2 = []
    num_blocks_in_layer3 = []
    num_blocks_in_layer4 = []
    for ik, k  in enumerate(new_state_dict.keys()):
        #resnets has four layers
        for l in np.arange(1,num_layers+1):
            if k.startswith('%s%d'%(modulestr,l)):
                vars()['num_blocks_in_layer%d'%l].extend([int(k.split('.')[blocknumsplit])])
               



    num_blocks_in_layer = []
    for l in np.arange(1,num_layers+1):

        num_blocks_in_layer.extend([max(vars()['num_blocks_in_layer%d'%l])]) 
    
    
    state_dict_layers = {}
    state_dict_others = {}
    
    for ik, k  in enumerate(new_state_dict.keys()):
        item = new_state_dict[k]
        for l in np.arange(1,num_layers+1):
            if k.startswith('%s'%(modulestr)):
                
                state_dict_layers.update({k:item})
            else:
                state_dict_others.update({k:item})

    state_dict_layers_new = {}
    for ik, k  in enumerate(state_dict_layers.keys()):
        item = state_dict_layers[k]
        for il, l in enumerate(np.arange(1,num_layers+1)):
            # print(k, '%s%d'%(modulestr,l))
            if k.startswith('%s%d'%(modulestr,l)):
                kswap = copy.deepcopy(k)
                block_num = int(k.split('.')[blocknumsplit])
                # print(range(num_blocks_in_a_layer[l]))
                list_block_num = list(range(num_blocks_in_layer[il]+1))
                list_block_num_reveresed = list_block_num[::-1]
                
                kswap = kswap.replace('.%d.'%block_num, '.%d.'%list_block_num_reveresed[block_num])  
                # print(l, k, kswap, item.shape)
                state_dict_layers_new.update({kswap:item})
    

    new_state_dict = {}
    new_state_dict.update(state_dict_layers_new)
    new_state_dict.update(state_dict_others)

    # # print(new_state_dict.keys())
    # if 'fc.weight' in state_dict_dest.keys():
    #     modulestr = modulestr.strip('layer')
    #     # print('aha!!!!!!')
    #     new_state_dict.update({modulestr+'fc.weight':state_dict_dest[modulestr+'fc.weight']})
    #     new_state_dict.update({modulestr+'fc.bias':state_dict_dest[modulestr+'fc.bias']})
    #     new_state_dict.update({modulestr+'biasfc':state_dict_dest[modulestr+'biasfc']})

    modulestr = modulestr.strip('layer')
    if modulestr+'fc.weight' in state_dict_dest.keys():    
        new_state_dict.update({modulestr+'fc.weight':state_dict_dest[modulestr+'fc.weight']})
    if modulestr+'fc.bias' in state_dict_dest.keys():
        new_state_dict.update({modulestr+'fc.bias':state_dict_dest[modulestr+'fc.bias']})
    if modulestr+'biasfc' in state_dict_dest.keys():
        new_state_dict.update({modulestr+'biasfc':state_dict_dest[modulestr+'biasfc']})

    # # only to leave bias1a and other biases for fixup intact
    # for k in new_state_dict.keys():
    #  if 'bias' in k:
    #      new_state_dict.update({k: state_dict_dest[k]})

    return new_state_dict



def toggle_state_dict_FFtoFB(state_dict, state_dict_dest):

    new_state_dict = {} #copy.deepcopy(state_dict_dest)
    already_added = []

    for ik, k  in enumerate(state_dict.keys()):

        item = state_dict[k]

        # Convolution's weight
        if ('bn' not in k) and ('weight' in k) and ('fc' not in k):

            if k not in already_added:
                

                if 'feedback' not in k:
                    item_dual = state_dict[k+'_feedback']
                    if 'upsample' in k:
                        k = k.replace('up','down')
                    elif 'downsample' in k:
                        k = k.replace('down', 'up')

                    new_state_dict.update({k+'_feedback':item})
                    new_state_dict.update({k:item_dual})
                

                else:
                    item_dual = state_dict[k.split('_')[0]]
                    if 'upsample' in k:
                        k = k.replace('up','down')
                    elif 'downsample' in k:
                        k = k.replace('down', 'up')
                    new_state_dict.update({k.split('_')[0]:item})
                    new_state_dict.update({k:item_dual})

        # fully connected's weight
        elif ('fc'in k) and ('weight' in k):
            if 'feedback' in k:
                new_state_dict.update({k.strip('_feedback'): torch.transpose(item, 0, 1)})
            else:
                new_state_dict.update({k+'_feedback': torch.transpose(item, 0, 1)})
        # other e.g. batchnorm    
        # else:
        #     new_state_dict.update({k:item})

        already_added.append(k)
    # print(new_state_dict.keys())
    state_dict_dest.update(new_state_dict)
    return state_dict_dest


def toggle_state_dict_BPtoYY(state_dict_BP, state_dict_YY):

    """ copies the forward of a BP net to backward of the YY net, preserving the
    backward of the YY net """

    new_state_dict = copy.deepcopy(state_dict_YY)
    already_added = []

    for ik, k  in enumerate(state_dict_BP.keys()):

        item = state_dict_BP[k]

        if ('bn' not in k) and ('weight' in k) and ('fc' not in k):

            if k not in already_added:
                
                if 'feedback' not in k:
                    if 'upsample' in k:
                        k = k.replace('up','down')
                    elif 'downsample' in k:
                        k = k.replace('down', 'up')

                    new_state_dict.update({k:item})
                

        elif ('fc'in k) and ('weight' in k):

            
            new_state_dict.update({k: torch.transpose(item, 0, 1)})
                
        else:
            new_state_dict.update({k:item})

        already_added.append(k)

    return new_state_dict

def toggle_state_dict_YYtoBP(state_dict_YY, state_dict_BP):

    """ copies the forward of a YY net to forward of the BP net
        (used to copy to the backward of BP but removed at April 29th)
    """

    new_state_dict = copy.deepcopy(state_dict_BP)

    for ik, k  in enumerate(state_dict_YY.keys()):

        item = state_dict_YY[k]

        if 'feedback' not in k:
            new_state_dict.update({k:item})
            # This part seems not necessary
            # if ('weight' in k) and ('bn' not in k):
            #     new_state_dict.update({k+'_feedback':item})

    return new_state_dict


def toggle_state_dict_SeeSaw(state_dict):

    new_state_dict = {}
    already_added = []

    for ik, k  in enumerate(state_dict.keys()):

        item = state_dict[k]

        if ('bn' not in k) and ('weight' in k):

            if k in already_added:
                pass

            else:
                if 'feedback' not in k:

                    new_state_dict.update({k+'_feedback':item})
                    new_state_dict.update({k:state_dict[k+'_feedback']})
                else:
                    new_state_dict.update({k.split('_')[0]:item})
                    new_state_dict.update({k:state_dict[k.split('_')[0]]})
            
            already_added.append(k)
        else:
            new_state_dict.update({k:item})


    return new_state_dict

def convert_FA_to_Symm(state_dict):


    new_state_dict = {}
    already_added = []

    for ik, k  in enumerate(state_dict.keys()):

        item = state_dict[k]
        if 'upsample' in k:
            k = k.replace('up','down')
        elif 'downsample' in k:
            k = k.replace('down', 'up')
        
      
        new_state_dict.update({k:item})
        if ('fc'in k) and ('weight' in k):
            new_state_dict.update({k:torch.transpose(item,0,1)})
        


    return new_state_dict


def toggle_weights(state_dict, state_dict_other_path):

    import copy

    state_dict_donor = copy.deepcopy(state_dict_other_path)

    import re


    for ctype in ['conv','tconv']:
        type_keys = [k for k in state_dict.keys() if k[0:len(ctype)]==ctype]
        if len(type_keys):
            conv_type = ctype
        else:
            conv_type_donor = ctype


    for layer_type in ['conv','tconv', 'fc'] :

        all_ltype_keys = [k for k in state_dict.keys() if k[0:len(layer_type)]==layer_type]
        layer_indices_type = np.unique([int(re.search('_(.+?).', k).group(1)) for k in state_dict.keys() if k in all_ltype_keys]).tolist()

        if len(all_ltype_keys):
            for ind in layer_indices_type:
                if layer_type == 'fc':
                    state_dict.update({'fc_%d.weight'%layer_indices_type[::-1][ind]: torch.transpose(state_dict_donor['fc_%d.weight_feedback'%ind],0,1)})
                    state_dict.update({'fc_%d.weight_feedback'%layer_indices_type[::-1][ind]: torch.transpose(state_dict_donor['fc_%d.weight'%ind],0,1)})
                elif layer_type == conv_type:
                    state_dict.update({'%s_%d.weight'%(conv_type, layer_indices_type[::-1][ind]): \
                                        state_dict_donor['%s_%d.weight_feedback'%(conv_type_donor,ind)]})

                    state_dict.update({'%s_%d.weight_feedback'%(conv_type, layer_indices_type[::-1][ind]): \
                                        state_dict_donor['%s_%d.weight'%(conv_type_donor,ind)]})

    return state_dict