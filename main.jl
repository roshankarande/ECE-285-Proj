using MLDatasets, Printf, Test
using SimpleChains

function get_data(split)
    x, y = MLDatasets.MNIST(split)[:]
    (reshape(x, 28 * 28 * 1, :), UInt32.(y .+ 1 ))
end

function display_loss(accuracy, loss)
    @printf("    training accuracy %.2f, loss %.4f\n", 100 * accuracy, loss)
end

xtrain, ytrain = get_data(:train);
xtest, ytest = get_data(:test);

σ(x) = max.(x,0)

model = SimpleChain(
            static(28 * 28),
            TurboDense(σ, 32),
            TurboDense(identity, 10))

begin

    num_image_classes = 10
    learning_rate = 3e-4
    num_epochs = 100

    loss = SimpleChains.add_loss(model, LogitCrossEntropyLoss(ytrain))
    G = SimpleChains.alloc_threaded_grad(loss)
    p = SimpleChains.init_params(model)

    opt = SimpleChains.ADAM(learning_rate)
    SimpleChains.train_batched!(G, p, loss, xtrain, opt, num_epochs)

    train_acc, train_loss = SimpleChains.accuracy_and_loss(loss, xtrain, ytrain, p)
    display_loss(train_acc, train_loss)

    test_acc, test_loss = SimpleChains.accuracy_and_loss(loss, xtest, ytest, p)
    display_loss(test_acc, test_loss)
end



x = xtrain[:,1];
size(p)
model(x,p)

W1, W2 = SimpleChains.weights(model, p)
b1, b2 = SimpleChains.biases(model, p)
W2*σ(W1*x + b1) + b2

@test model(x,p) ≈  W2*σ(W1*x + b1) + b2


using FileIO

save( "params.jld2", Dict("W1" => Float32.(W1), "W2" => Float32.(W2), "b1" => Float32.(b1), "b2" => Float32.(b2)))

params = load("params.jld2")