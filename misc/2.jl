import MLDatasets, SimpleChains

function get_data()
    xtrain, ytrain = MNIST(:train)[:]
    xtest, ytest = MNIST(split=:test)[:]

    (
        (reshape(xtrain, 28 * 28 * 1, :), UInt32.(ytrain .+ 1)),
        (reshape(xtest, 28 * 28 * 1, :), UInt32.(ytest .+ 1)),
    )
end

(xtrain, ytrain), (xtest, ytest) = get_data();


lenet = SimpleChain(
    (static(28 * 28)),
    TurboDense(SimpleChains.relu, 32),
    TurboDense(identity, 10),
)


@time p = SimpleChains.init_params(lenet);

@time lenet(xtrain, p)

lenetloss = SimpleChains.add_loss(lenet, LogitCrossEntropyLoss(ytrain));

g = similar(p);
@time valgrad!(g, lenetloss, xtrain, p)

G = SimpleChains.alloc_threaded_grad(lenetloss);

@time SimpleChains.train_batched!(G, p, lenetloss, xtrain, SimpleChains.ADAM(3e-4), 10);

SimpleChains.accuracy_and_loss(lenetloss, xtrain, p),
SimpleChains.accuracy_and_loss(lenetloss, xtest, ytest, p)

# SimpleChains.init_params!(lenet, p);
@time SimpleChains.train_batched!(G, p, lenetloss, xtrain, SimpleChains.ADAM(3e-4), 10);
SimpleChains.accuracy_and_loss(lenetloss, xtrain, p),
SimpleChains.accuracy_and_loss(lenetloss, xtest, ytest, p)
