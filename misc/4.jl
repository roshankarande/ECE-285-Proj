using MLDatasets, Printf
using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using Statistics, Random
using CUDA

num_image_classes = 10
learning_rate = 3e-4
num_epochs = 10


function get_data(split)
    x, y = MLDatasets.MNIST(split)[:]
    (reshape(x, 28 * 28 * 1, :), UInt32.(y .+ 1))
end

function display_loss(accuracy, loss)
    @printf("    training accuracy %.2f, loss %.4f\n", 100 * accuracy, loss)
end

xtrain, ytrain = get_data(:train);
xtest, ytest = get_data(:test);


use_cuda = CUDA.functional()
batchsize = 0
device = if use_cuda
    @info "Flux training on GPU"
    batchsize = 2048
    gpu
else
    @info "Flux training on CPU"
    batchsize = 96 * Threads.nthreads()
    cpu
end

function create_loader(x, y, batch_size, shuffle)
    y = onehotbatch(y, 1:num_image_classes)
    DataLoader((device(x), device(y)), batchsize=batch_size, shuffle=shuffle)
end

train_loader = create_loader(xtrain, ytrain, batchsize, true)
test_loader = create_loader(xtest, ytest, batchsize, false)

function flux_model()
    return Chain(
        Dense(28 * 28, 32, Flux.relu),
        Dense(32, 10)
    ) |> device
end

flux_loss(ŷ, y) = logitcrossentropy(ŷ, y)

function eval_accuracy_loss(loader, model, device)
    l = 0.0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        l += flux_loss(ŷ, y) * size(x)[end]
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))

        ntot += size(x)[end]
    end
    return (acc=acc / ntot, loss=l / ntot)
end

function train!(model, train_loader, opt)
    ps = Flux.params(model)
    for _ = 1:num_epochs
        for (x, y) in train_loader
            x = device(x)
            y = device(y)
            gs = Flux.gradient(ps) do
                ŷ = model(x)
                flux_loss(ŷ, y)
            end
            Flux.Optimise.update!(opt, ps, gs)
        end
    end
end

@time "  create model" model = flux_model()
opt = ADAM(learning_rate)
@time "  train $num_epochs epochs" train!(model, train_loader, opt)
@time "  compute training loss" train_acc, train_loss = eval_accuracy_loss(test_loader, model, device)
display_loss(train_acc, train_loss)
@time "  compute test loss" test_acc, test_loss = eval_accuracy_loss(train_loader, model, device)
display_loss(test_acc, test_loss)
println("------------------------------------")


model = flux_model()

count = 0
for layer in model
    count += sum(length, Flux.params(layer))
    # println(count)
    # println(Flux.params(layer))
end

for layer in model
    println(size(layer.weight), size(layer.bias))
end


# ----------------------------------

x = xtrain[:,1] |> device;
W1 = model[1].weight 
b1 = model[1].bias
W2 = model[2].weight 
b2 = model[2].bias 

model(x)

W2*Flux.relu(W1*x + b1) + b2