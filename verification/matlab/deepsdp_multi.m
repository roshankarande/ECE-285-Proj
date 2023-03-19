function [bound, time,status] =  deepsdp_multi(net,x_min,x_max,label,target,options)

% DeepSDP from "Mahyar Fazlyab, Manfred Morari, George J. Pappas, "Safety Verification and Robustness Analysis of Neural Networks via Quadratic Constraints and Semidefinite Programming"
% DeepSDP is the implementation of dual sdp relaxation method
version = '1.0';

language = options.language;
solver = options.solver;
verbose = options.verbose;

weights = net.weights;
biases = net.biases;
activation = net.activation;
dims = net.dims;

[Y_min,Y_max,X_min,X_max,~,~] = net.interval_arithmetic(x_min,x_max);

% input dimension
dim_in = dims(1);

% output dimension
dim_out = dims(end);

dim_last_hidden = dims(end-1);

% total number of neurons
num_neurons = sum(dims(2:end-1));


%% Construct Min
tau = sdpvar(dim_in,1);
constraints = [tau(:)>=0];

P = [-2*diag(tau) diag(tau)*(x_min+x_max);(x_min+x_max).'*diag(tau) -2*x_min.'*diag(tau)*x_max];
tmp = ([eye(dim_in) zeros(dim_in,num_neurons+1);zeros(1,dim_in+num_neurons) 1]);
Min = tmp.'*P*tmp;

%% QC for activation functions
if(strcmp(activation,'relu'))
    nu = sdpvar(num_neurons,1);
        
    lambda = sdpvar(num_neurons,1);
        
    eta = sdpvar(num_neurons,1);
        
    D = diag(sdpvar(num_neurons,1));

    constraints = [constraints, nu(:)>=0, nu(:)>=0, eta(:)>=0, eta(:)>=0, D(:)>=0];

    alpha_param = zeros(num_neurons,1);

    beta_param = ones(num_neurons,1);

    Q11 = -2*diag(alpha_param.*beta_param)*(diag(lambda));
    Q12 = diag(alpha_param+beta_param)*(diag(lambda));
    Q13 = -nu;
    Q22 = -2*diag(lambda)-2*D;
    Q23 = nu+eta+D*(X_min+X_max);
    Q33 = -2*X_min'*D*X_max;

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
mosek_options = sdpsettings('solver',solver,'verbose',verbose, 'mosek.MSK_IPAR_BI_IGNORE_MAX_ITER', 1, 'mosek.MSK_IPAR_BI_IGNORE_NUM_ERROR', 1);
disp("Solving problem -- DeepSDP")
out = optimize(constraints,obj,mosek_options);
    
bound = value(obj);
if options.mode == 2 & label == 2
    bound = -bound;
end
time= out.solvertime;
status = out.info;

message = ['method: DeepSDP ', version,'| solver: ', solver, '| bound: ', num2str(bound,'%.5f'), '| solvetime: ', num2str(time,'%.3f'), '| status: ', status];

disp(message);

end
