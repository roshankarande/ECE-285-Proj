function [subi, subj, subk, subl, val, blc, buc, ret_index] = construct_constriants(constraints_index, layer_index, dims, W_i, b_i, l_i, u_i, y_min, y_max)
% constraints_index: global index of constraints
% layer_index: the layer index i of the neural network
% type: there are 4 different types of constraints corresponding to each
% layer
% W_i: weights of the neural network in layer i
% b_i: bias of the neural network in layer i
% l_i: the lower bound of last layer i-1
% u_i: the upper bound of last layer i-1

% pre-allocate the space
preallocate_space = dims(layer_index + 1);
%subi = zeros(1, preallocate_space * preallocate_space);
%subj = ones(1, preallocate_space * preallocate_space);
%subk = zeros(1, preallocate_space * preallocate_space);
%subl = zeros(1, preallocate_space * preallocate_space);
%val = zeros(1, preallocate_space * preallocate_space);
%blc = zeros(1, preallocate_space * 4);
%buc = zeros(1, preallocate_space * 4);


subi = [];
subj = [];
subk = [];
subl = [];
val = [];
blc = [];
buc = [];

dim_i = dims(layer_index + 1);
dim_prev = dims(layer_index);

prev_prev_dims_sum = sum(dims(1:layer_index - 1));
prev_dims_sum = sum(dims(1:layer_index));
index = 1; % local index of the constraints

blc_buc_index = 1;

if layer_index == length(dims(1:end-1))
    % type-4 constraints
    row_pointer = prev_prev_dims_sum;
    for j = 1: dim_prev
        l_i_j = l_i(j);
        u_i_j = u_i(j);
        blc(blc_buc_index) = l_i_j * u_i_j;
        buc(blc_buc_index) = inf;
        blc_buc_index = blc_buc_index + 1;
        subi(index) = constraints_index;
        subj(index) = 1;
        subk(index) = row_pointer + 2;
        subl(index) = 1;
        val(index) = 0.5 * (l_i_j + u_i_j);
        index = index + 1;
        
        subi(index) = constraints_index;
        subj(index) = 1;
        subk(index) = row_pointer + 2;
        subl(index) = row_pointer + 2;
        val(index) = -1;
        index = index + 1;
        row_pointer = row_pointer + 1;
        
        constraints_index = constraints_index + 1;
    end


    subi(index) = constraints_index;
    subj(index) = 1;
    subk(index) = 1;
    subl(index) = 1;
    val(index) = 1;
    blc(blc_buc_index) = 1;
    buc(blc_buc_index) = 1;
    constraints_index = constraints_index + 1;
else
    % type-1 constraints
    for j = 1:dim_i
        W_i_j = W_i(j,:); % the j-th row of W_i
        pointer = prev_prev_dims_sum;
        blc(blc_buc_index) = b_i(j);
        buc(blc_buc_index) = inf;
        blc_buc_index = blc_buc_index + 1;
        for k = 1:(dim_prev + 1)
            subi(index) = constraints_index;
            subj(index) = 1;
            if k <= dim_prev
                subk(index) = pointer + 2;
            else
                subk(index) = j + prev_dims_sum + 1;
            end
            
            subl(index) = 1;
            if k <= dim_prev
                val(index) = -0.5 * W_i_j(k);
            else
                val(index) = 0.5;
            end
            index = index + 1;
            pointer = pointer + 1;
        end
        constraints_index = constraints_index + 1;
    end
    
    % type-2 constraints
    for j = 1:dim_i
        pointer = prev_dims_sum;
        blc(blc_buc_index) = 0;
        buc(blc_buc_index) = inf;
        blc_buc_index = blc_buc_index + 1;
        subi(index) = constraints_index;
        subj(index) = 1;
        subk(index) = pointer + j + 1;
        subl(index) = 1;
        val(index) = 1;
        index = index + 1;
        constraints_index = constraints_index + 1;
    end
    
    % type-3 constraints
    row_pointer = prev_dims_sum;
    for j = 1: dim_i
        W_i_j = W_i(j,:); % the j-th row of W_i
        blc(blc_buc_index) = -inf;
        buc(blc_buc_index) = 0;
        blc_buc_index = blc_buc_index + 1;
        subi(index) = constraints_index;
        subj(index) = 1;
        subk(index) = row_pointer + 2;
        subl(index) = 1;
        val(index) = -b_i(j); %
        index = index + 1;
        col_pointer = prev_prev_dims_sum;
        for k = 1:(dim_prev)
            subi(index) = constraints_index;
            subj(index) = 1;
            subk(index) = row_pointer + 2;
            subl(index) = col_pointer + 2;
            val(index) = -W_i_j(k);
            col_pointer = col_pointer + 1;
            index = index + 1;
        end
        subi(index) = constraints_index;
        subj(index) = 1;
        subk(index) = row_pointer + 2;
        subl(index) = row_pointer + 2;
        val(index) = 2;
        row_pointer = row_pointer + 1;
        constraints_index = constraints_index + 1;
        index = index + 1;
    end
    
    % type-4 constraints
    row_pointer = prev_prev_dims_sum;
    for j = 1: dim_prev
        l_i_j = l_i(j);
        u_i_j = u_i(j);
        blc(blc_buc_index) = l_i_j * u_i_j;
        buc(blc_buc_index) = inf;
        blc_buc_index = blc_buc_index + 1;
        subi(index) = constraints_index;
        subj(index) = 1;
        subk(index) = row_pointer + 2;
        subl(index) = 1;
        val(index) = 0.5 * (l_i_j + u_i_j);
        index = index + 1;
        
        subi(index) = constraints_index;
        subj(index) = 1;
        subk(index) = row_pointer + 2;
        subl(index) = row_pointer + 2;
        val(index) = -1;
        index = index + 1;
        row_pointer = row_pointer + 1;
        
        constraints_index = constraints_index + 1;
    end

    % linear cut constraints
    for j = 1:dim_i
        k_i = (relu(y_max(j)) - relu(y_min(j))) / (y_max(j) - y_min(j));
        W_i_j = W_i(j,:); % the j-th row of W_i
        pointer = prev_prev_dims_sum;
        blc(blc_buc_index) = -inf;
        buc(blc_buc_index) = k_i * (b_i(j) - y_min(j)) + relu(y_min(j));
        blc_buc_index = blc_buc_index + 1;

        for k = 1:(dim_prev + 1)
            subi(index) = constraints_index;
            subj(index) = 1;
            if k <= dim_prev
                subk(index) = pointer + 2;
            else
                subk(index) = j + prev_dims_sum + 1;
            end

            subl(index) = 1;
            if k <= dim_prev
                val(index) = -0.5 * k_i * W_i_j(k);
            else
                val(index) = 0.5;
            end
            index = index + 1;
            pointer = pointer + 1;
        end
        constraints_index = constraints_index + 1;
    end
end
ret_index = constraints_index;
end

function [y] = relu(x)
    y = max(x, 0);
end