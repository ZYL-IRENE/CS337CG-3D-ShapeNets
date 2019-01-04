function model = run_pretrain()
    % Layer-wise pretraining.
    
    rng('shuffle');
    kernels;
    
    param = [];
    param.debug = 0;
    param.volume_size = 21;%13;%21;%11;%21;%24;
    param.pad_size = 0;%4;%5;%3;
    data_size = param.volume_size + 2 * param.pad_size;
    
    %% load data
    param.data_path = '~/Dropbox/3dprior/data_CDBM_semantic_v3_semantic_abstract_da';%data_CDBM_semantic_v3_semantic';%data_CDBM_synthetic';%data_CDBM_semantic';
    load([param.data_path '/classlabels.mat']);
    param.classnames = unique_semantic_labels;%{'good','bad'};
    
    % param.data_path = '~/Dropbox/3dprior/data_CDBM_synthetic';%data_CDBM_semantic_v3_semantic';%data_CDBM_synthetic';%data_CDBM_semantic';
    % param.classnames = {'good', 'bad'};
    
    
    %param.classnames = {'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet'};
    % param.classnames = {'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet', ...
    %            'airplane', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'cone', 'cup', 'curtain', 'door', ...
    %            'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'person', 'piano', 'plant', ...
    %            'radio', 'range_hood', 'sink', 'stairs', 'stool', 'tent', 'tv_stand', 'vase', 'wardrobe', 'xbox'};
    param.classes = length(param.classnames)
    data_list = read_data_list(param.data_path, param.classnames, data_size, 'train', param.debug);
    % load('bad.mat','data_list');
    % save('semantic_data.mat','data_list','unique_semantic_labels');
    %%
    % idx=100
    % data_list{2}(idx).filename
    % fn=(data_list{2}(idx).filename)
    % load(fn)
    % instance4D(1,:,:,:)=instance;
    % show_1_sample(instance4D(1,:,:,:))
    % save('part_bad_stickout6.mat','instance');
    
    %%
    % kill = [];
    % for i=1:1:275
    %     data_list{1}(i).filename
    %     fn=(data_list{1}(i).filename)
    %     load(fn)
    %     instance4D(1,:,:,:)=instance;
    %     show_1_sample(instance4D(1,:,:,:))
    %     instance4D(1,:,:,:)=instance;
    %     if checkKey()==0
    %         kill = [kill ; i];
    %     end
    %     %pause
    % end
    % % data_list{1}(kill)=[];
    % save('good.mat','data_list');
    % kill=[];
    % for i=1:1:296
    %     data_list{2}(i).filename
    %     fn=(data_list{2}(i).filename)
    %     load(fn)
    %     instance4D(1,:,:,:)=instance;
    %     show_1_sample(instance4D(1,:,:,:))
    %     instance4D(1,:,:,:)=instance;
    %     if checkKey()==0
    %         kill = [kill ; i];
    %     end
    %     %pause
    % end
    % data_list{2}(kill)=[];
    
    
    param.network = {
        struct('type', 'input');
        struct('type', 'convolution', 'outputMaps', 48, 'kernelSize', 7, 'actFun', 'sigmoid', 'stride', 1);
        struct('type', 'convolution', 'outputMaps', 160, 'kernelSize', 5, 'actFun', 'sigmoid', 'stride', 2);
        struct('type', 'convolution', 'outputMaps', 512, 'kernelSize', 4, 'actFun', 'sigmoid', 'stride', 1);
        struct('type', 'fullconnected', 'size', 1200, 'actFun', 'sigmoid');%64, 'actFun', 'sigmoid');
        struct('type', 'fullconnected', 'size', 4000, 'actFun', 'sigmoid');%80, 'actFun', 'sigmoid');
    };
    % param.network = {
    %     struct('type', 'input');
    %     struct('type', 'convolution', 'outputMaps', 48, 'kernelSize', 6, 'actFun', 'sigmoid', 'stride', 2);
    %     struct('type', 'convolution', 'outputMaps', 160, 'kernelSize', 5, 'actFun', 'sigmoid', 'stride', 2);
    %     struct('type', 'convolution', 'outputMaps', 512, 'kernelSize', 4, 'actFun', 'sigmoid', 'stride', 1);
    %     struct('type', 'fullconnected', 'size', 1200, 'actFun', 'sigmoid');
    %     struct('type', 'fullconnected', 'size', 4000, 'actFun', 'sigmoid');
    % };
    
    
    % This is to duplicate the labels for the final RBM in order to enforce the
    % label training.
    param.duplicate = 10;
    param.validation = 1;
    param.data_size = [data_size, data_size, data_size, 1];
    
    model = initialize_cdbn(param);
    
    fprintf('\nmodel initialzation completed!\n\n');
    param = [];
    param.layer = 2;
    param.epochs = 150;
    param.lr = 0.015;
    param.weight_decay = 1e-5;
    param.momentum = [0.5, 0.9];
    param.kPCD = 1;
    param.persistant = 0;
    param.batch_size = 32;
    param.sparse_damping = 0;
    param.sparse_target = 0.01;
    param.sparse_cost = 0.03;
    [model] = crbm2(model, data_list, param);
    save('model30_l2','model');
    % load('model30_l2','model');
    
    param = [];
    param.layer = 3;
    param.epochs = 400;
    param.lr = 0.003;
    param.weight_decay = 1e-5;
    param.momentum = [0.5, 0.9];
    param.kPCD = 1;
    param.persistant = 0;
    param.batch_size = 32;
    param.sparse_damping = 0;
    param.sparse_target = 0.05;
    param.sparse_cost = 0.1;
    [model] = crbm(model, data_list, param);
    save('model30_l3','model');
    % load('model30_l3','model');
    
    
    layer_idx = 5;
    if model.numLayer == 6
        param = [];
        param.layer = 4;
        param.epochs = 600;
        param.lr = 0.002;
        param.weight_decay = 1e-5;
        param.momentum = [0.5, 0.9];
        param.kPCD = 1;
        param.persistant = 0;
        param.batch_size = 32;
        param.sparse_damping = 0;
        param.sparse_target = 0;
        param.sparse_cost = 0;
        [model] = crbm(model, data_list, param);
        save('model30_l4','model');
    %      load('model30_l4','model');
    else
        layer_idx=4;
    end
    
    [hidden_prob_h4, train_label] = propagate_data(model, data_list, layer_idx);%5);
    
    param = [];
    param.layer = layer_idx;%5;
    param.epochs = 600;
    param.lr = 0.002;
    param.weight_decay = 1e-5;
    param.momentum = [0.5, 0.9];
    param.kPCD = 1;
    param.persistant = 0;
    param.batch_size = 32;
    param.sparse_damping = 0;
    param.sparse_target = 0;
    param.sparse_cost = 0;
    new_list = balance_data(data_list,param.batch_size);
    hidden_prob_h4 = reshape(hidden_prob_h4, size(hidden_prob_h4,1),[]);
    % [model, hidden_prob_h5] = rbm(model, new_list, hidden_prob_h4, param);
    % save('model30_l5','model');
    load('model30_l5','model');
    param.epochs=1;
    [model, hidden_prob_h5] = rbm(model, new_list, hidden_prob_h4, param);
    
    
    param = [];
    param.layer = layer_idx+1;%6;
    param.epochs = 600;
    param.lr = 0.0003;
    param.weight_decay = 1e-5;
    param.momentum = [0.5, 0.5];
    param.kPCD = 1;
    param.persistant = 1;
    param.batch_size = 32;
    param.sparse_damping = 0;
    param.sparse_target = 0;
    param.sparse_cost = 0;
    new_list = balance_data(data_list,param.batch_size);
    %[model] = rbm(model, new_list, [train_label, hidden_prob_h5], param);
    hidden_prob_h5 = reshape(hidden_prob_h5, size(hidden_prob_h5,1),[]);
    [model] = rbm_last(model, new_list, [train_label, hidden_prob_h5], param);
    save('model30_l6','model');
    
    %% train last layer (output layer)
    % param = [];
    % param.layer = layer_idx+2;%6;
    % param.epochs = 2000;
    % param.lr = 0.0003;
    % param.weight_decay = 1e-5;
    % param.momentum = [0.5, 0.5];
    % param.kPCD = 1;
    % param.persistant = 1;
    % param.batch_size = 32;
    % param.sparse_damping = 0;
    % param.sparse_target = 0;
    % param.sparse_cost = 0;
    % new_list = balance_data(data_list,param.batch_size);
    % [model] = rbm_last(model, new_list, [train_label, hidden_prob_h6], param);
    % save('model30_l7','model');
    
    for l = 2 : length(model.layers)
        model.layers{l} = rmfield(model.layers{l},'grdw');
        model.layers{l} = rmfield(model.layers{l},'grdb');
        model.layers{l} = rmfield(model.layers{l},'grdc');
    end
    
    save('pretrained_model','model');