using MLDatasets, SimpleChains
using Printf

# Get MNIST data
function get_data(split)
    x, y = MLDatasets.MNIST(split)[:]
    (reshape(x, 28 * 28 * 1, :), UInt32.(y .+ 1))
end

function display_loss(accuracy, loss)
    @printf("    training accuracy %.2f, loss %.4f\n", 100 * accuracy, loss)
end

begin
    xtrain, ytrain = get_data(:train)
    img_size = Base.front(size(xtrain))
    xtest, ytest = get_data(:test)
end
# Training parameters

num_image_classes = 10
learning_rate = 3e-4
num_epochs = 100


# SimpleChains implementation
begin

    lenet = SimpleChain(
        (static(28 * 28)),
        TurboDense(SimpleChains.relu, 32),
        TurboDense(identity, 10),
    )

    lenetloss = SimpleChains.add_loss(lenet, LogitCrossEntropyLoss(ytrain))

    for i in 1:2
        println("SimpleChains run #$i")
        @time "  gradient buffer allocation" G = SimpleChains.alloc_threaded_grad(lenetloss)
        @time "  parameter initialization" p = SimpleChains.init_params(lenet)
        #@time "  forward pass" lenet(xtrain, p)

        #g = similar(p);
        #@time "  valgrad!" valgrad!(g, lenetloss, xtrain, p)

        opt = SimpleChains.ADAM(learning_rate)
        @time "  train $(num_epochs) epochs" SimpleChains.train_batched!(G, p, lenetloss, xtrain, opt, num_epochs)

        @time "  compute training accuracy and loss" train_acc, train_loss = SimpleChains.accuracy_and_loss(lenetloss, xtrain, ytrain, p)
        display_loss(train_acc, train_loss)

        @time "  compute test accuracy and loss" test_acc, test_loss = SimpleChains.accuracy_and_loss(lenetloss, xtest, ytest, p)
        display_loss(test_acc, test_loss)
    end
end