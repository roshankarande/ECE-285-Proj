function [bound, time,status] =  sdpnet(net,x_min,x_max,label,target,num, options)
% SDPNET is built based on the dual SDP method
version = '1.0';

disp(options);
language = options.language;
solver = options.solver;
verbose = options.verbose;

weights = net.weights;
biases = net.biases;
activation = net.activation;
dims = net.dims;


num_hidden_layers = length(dims)-2;

k = 1;
X_min{k} = x_min;
X_max{k} = x_max;

% interval propagation
if(options.bounds ~= "crown")
    for k=1:num_hidden_layers+1
        Y_min{k} = max(net.weights{k},0)*X_min{k}+min(net.weights{k},0)*X_max{k}+net.biases{k}(:);
        Y_max{k} = min(net.weights{k},0)*X_min{k}+max(net.weights{k},0)*X_max{k}+net.biases{k}(:);
        
        if(k<=num_hidden_layers)
            X_min{k+1} = max(Y_min{k},0);
            X_max{k+1} = max(Y_max{k},0);
        end
    end
end

% CROWN
if(options.bounds == "crown")
    for k=1:num_hidden_layers
        Y_max{k} = double(load("y_max").y_max{k}');
        Y_min{k} = double(load("y_min").y_min{k}');
        X_max{k+1} = double(load("x_max").x_max{k}');
        X_min{k+1} = double(load("x_min").x_min{k}');
    end
end

% input dimension
dim_in = dims(1);
% output dimension
dim_out = dims(end);

% remove vacuous neurons
for i = 1: num_hidden_layers
    In_i = find(Y_max{i} <= 0);
    cp_In_i = setdiff(1: dims(i + 1),In_i);
    weights{i} = weights{i}(cp_In_i, :);
    weights{i + 1} = weights{i + 1}(:, cp_In_i);
    biases{i} = biases{i}(cp_In_i);
    X_min{i + 1} = X_min{i +1}(cp_In_i);
    X_max{i + 1} = X_max{i +1}(cp_In_i);
    Y_min{i} = Y_min{i}(cp_In_i);
    Y_max{i} = Y_max{i}(cp_In_i);
    dims(i + 1) = dims(i + 1) - length(In_i);
end
disp("after removing vacuous neurons: ")
disp(dims);

X_min = cat(1, X_min{2:end});
X_max = cat(1, X_max{2:end});
if options.mode == 1
    Y_min = cat(1, Y_min{1: end});
    Y_max = cat(1, Y_max{1:end});
end
if options.mode == 2
    Y_min = cat(1, Y_min{1: end - 1});
    Y_max = cat(1, Y_max{1:end - 1});
end


dim_last_hidden = dims(end-1);

% total number of neurons
num_neurons = sum(dims(2:end-1));

Ip = find(Y_min>0);
In = find(Y_max<0);
Ipn = setdiff(1:num_neurons,union(Ip,In));

%% Construct Min
tau = sdpvar(dim_in,1);
constraints = [tau(:)>=0];
P = [-2*diag(tau) diag(tau)*(x_min+x_max);(x_min+x_max).'*diag(tau) -2*x_min.'*diag(tau)*x_max];
tmp = ([eye(dim_in) zeros(dim_in,num_neurons+1);zeros(1,dim_in+num_neurons) 1]);
Min = tmp.'*P*tmp;

%% QC for activation functions
if(strcmp(activation,'relu'))
    T = zeros(num_neurons);
    II = eye(num_neurons);
    C = [];
    % if neurons number is greater than 50, then only choose 50 neurons
    % randomly
    if num_neurons >= num
        rand_i = randi([1, num_neurons], 1, num);
        C = nchoosek(rand_i, 2);
    else
        C = nchoosek(1:num_neurons, 2);
    end
    
    m = size(C,1);
    
    if(m>0)
        
        zeta = sdpvar(m,1);
        constraints = [constraints,zeta>=0];
        
        E = II(:,C(:,1))-II(:,C(:,2));
        T = E*diag(zeta)*E';
    end
    
    
    
    nu = sdpvar(num_neurons,1);
    
    lambda = sdpvar(num_neurons,1);
    
    eta = sdpvar(num_neurons,1);
    
    D = diag(sdpvar(num_neurons,1));
    
    delta = diag(sdpvar(num_neurons,1));
    
    constraints = [constraints, nu(In)>=0, nu(Ipn)>=0, eta(Ip)>=0, eta(Ipn)>=0, D(:)>=0, delta(Ipn,Ipn) >= 0, delta(Ip,Ip) == 0, delta(In,In) == 0];
    
    
    
    alpha_param = zeros(num_neurons,1);
    alpha_param(Ip)=1;
    
    
    
    beta_param = ones(num_neurons,1);
    original_beta = beta_param;
    beta_param(In) = 0;
    
    disp(size(Y_max));
    disp(size(delta));
    
    Q11 = -2*diag(alpha_param.*beta_param)*(diag(lambda));
    Q12 = diag(alpha_param+beta_param)*(diag(lambda))+T;
    Q13 = -nu + delta * (Y_max);
    Q22 = -2*diag(lambda)-2*D-2*T;
    Q23 = nu+eta+D *(X_min+X_max) + delta * (Y_min - Y_max);
    Q33 = -2*X_min'*D*X_max - 2*Y_min'*delta*Y_max;
    
    Q = [Q11 Q12 Q13; Q12.' Q22 Q23;Q13.' Q23.' Q33];
else
    error('The method deep sdp is currently supported for ReLU activation functions only.');
end

%% Construct Mmid
A = ([blkdiag(weights{1:end-1}) zeros(num_neurons,dim_last_hidden)]);
B = ([zeros(num_neurons,dim_in) eye(num_neurons)]);
bb = cat(1,biases{1:end-1});
CM_mid = ([A bb;B zeros(size(B,1),1);zeros(1,size(B,2)) 1]);
Mmid = CM_mid.'*Q*CM_mid;

%% Construct Mout
b = sdpvar(1);


obj = b;

c = zeros(dim_out,1);
% classification problem
if options.mode == 1
    c(label) = -1;
    c(target) = 1;
end
% prediction problem
if options.mode == 2
    if label == 1
        c(1) = 1;
    end
    if label == 2
        c(1) = -1;
    end
end

S = [zeros(dim_out,dim_out) c;c' -2*b];
tmp = ([zeros(dim_out,dim_in+num_neurons-dim_last_hidden) weights{end} biases{end};zeros(1,dim_in+num_neurons) 1]);
Mout = tmp.'*S*tmp;

%% solve SDP

constraints = [constraints, Min+Mmid+Mout<=0];
sdp_options = sdpsettings('solver',solver,'verbose',verbose);
disp("Solving problem -- SDPNET")
out = optimize(constraints,obj,sdp_options);
bound = value(obj);
if options.mode == 2 & label == 2
    bound = -bound;
end
time= out.solvertime;
status = out.info;


message = ['method: SDPNET ', version,'| solver: ', solver, '| bound: ', num2str(bound,'%.5f'), '| solvetime: ', num2str(time,'%.3f'), '| status: ', status];



disp(message);

end
