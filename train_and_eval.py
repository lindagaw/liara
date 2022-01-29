import torch.nn as nn
import torch.optim as optim
from misc import make_variable

batch_size = 128

image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 150
lr = 0.0002
beta1 = 0.5
ngpu = 4

def train(classifier, data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers

    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=(beta1, 0.999))


    ####################
    # 2. train network #
    ####################

    for epoch in range(num_epochs):
        for step, (images, labels) in enumerate(data_loader):
            # make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(images)
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % 10 == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              10,
                              step + 1,
                              len(data_loader),
                              loss.data[0]))


    return classifier


def eval_src(classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers

    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels)

        preds = classifier(images)
        loss += criterion(preds, labels).data[0]

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
