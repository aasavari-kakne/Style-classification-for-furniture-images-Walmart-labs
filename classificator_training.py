import torch
from torch.utils.data import DataLoader
from imagenet_dataset import get_imagenet_datasets
from helper_functions import get_patch_tensor_from_image_batch, inspect_model, write_csv_stats,plot_confusion_matrix

import os
import numpy as np
from sklearn.metrics import confusion_matrix
from imagenet_dataset import ClASS_LIST

def run_classificator(args, res_classificator_model, res_encoder_model, models_store_path):

    print("RUNNING CLASSIFICATOR TRAINING")
    dataset_train, dataset_test, dataset_val = get_imagenet_datasets(args.image_folder, num_classes = args.num_classes, train_split = 1.0, random_seed = 42,mode="classificator")

    CLASS_TUPLE=list(ClASS_LIST)

    stats_csv_path = os.path.join(models_store_path, "classification_stats.csv")
    train_csv_path = os.path.join(models_store_path, "train_batch_stats.csv")
    test_csv_path = os.path.join(models_store_path, "test_batch_stats.csv")

    EPOCHS = 1
    LABELS_PER_CLASS = 25 # not used yet

    data_loader_train = DataLoader(dataset_train, args.sub_batch_size, shuffle = True)
    data_loader_test = DataLoader(dataset_test, args.sub_batch_size, shuffle = True)

    NUM_TRAIN_SAMPLES = dataset_train.get_number_of_samples()
    NUM_TEST_SAMPLES = dataset_test.get_number_of_samples()

    # params = list(res_classificator_model.parameters()) + list(res_encoder_model.parameters())
    '''encoder is trained as well???'''
    optimizer_enc = torch.optim.Adam(params = res_encoder_model.parameters(), lr = 0.000001) # Train encoder slower than the classifier layers
    optimizer_cls = torch.optim.Adam(params = res_classificator_model.parameters(), lr = 0.00001)

    best_epoch_test_loss = 1e10
#    all_preds = torch.tensor([]).to(args.device)
#    all_inds=torch.tensor([]).to(args.device)

    for epoch in range(EPOCHS):

        sub_batches_processed = 0

        epoch_train_true_positives = 0.0
        epoch_train_loss = 0.0
        epoch_train_losses = []

        batch_train_loss = 0.0
        batch_train_true_positives = 0.0

        epoch_test_true_positives = 0.0
        epoch_test_loss = 0.0

        epoch_test_losses = []

        for batch in data_loader_train:
            all_preds = torch.tensor([]).to(args.device)
            all_inds=torch.tensor([]).to(args.device)
        

            img_batch = batch['image'].to(args.device)

            patch_batch = get_patch_tensor_from_image_batch(img_batch)
            patches_encoded = res_encoder_model.forward(patch_batch)

            patches_encoded = patches_encoded.view(img_batch.shape[0], 7,7,-1)
            #patches_encoded = patches_encoded.permute(0,3,1,2)
            patches_encoded = patches_encoded.permute(0,3,1,2).contiguous()

            classes = batch['cls'].to(args.device)

            y_one_hot = torch.zeros(img_batch.shape[0], args.num_classes).to(args.device)
            y_one_hot = y_one_hot.scatter_(1, classes.unsqueeze(dim=1), 1)

            labels = batch['class_name']

            pred = res_classificator_model.forward(patches_encoded)
            loss = torch.sum(-y_one_hot * torch.log(pred))
            epoch_train_losses.append(loss.detach().to('cpu').numpy())
            epoch_train_loss += loss.detach().to('cpu').numpy()
            batch_train_loss += loss.detach().to('cpu').numpy()

            epoch_train_true_positives += torch.sum(pred.argmax(dim=1) == classes)
            batch_train_true_positives += torch.sum(pred.argmax(dim=1) == classes)

            loss.backward()
            sub_batches_processed += img_batch.shape[0]

            if sub_batches_processed >= args.batch_size:

                optimizer_enc.step()
                optimizer_cls.step()

                optimizer_enc.zero_grad()
                optimizer_cls.zero_grad()

                sub_batches_processed = 0

                batch_train_accuracy = float(batch_train_true_positives) / float(args.batch_size)

                print(f"Training loss of batch is {batch_train_loss}")
                print(f"Accuracy of batch is {batch_train_accuracy}")
                train_stats = dict(
                    #iteration=t*(epoch+1)
                    batch_train_loss = batch_train_loss,
                    batch_train_accuracy = batch_train_accuracy
                )

                print("Writing dict {} into file {}".format(train_stats,train_csv_path))
                write_csv_stats(train_csv_path, train_stats)

                batch_train_loss = 0.0
                batch_train_true_positives = 0.0
                
                with torch.no_grad():
                    batch_test_acc=0
                    batch_test_loss=0
                    batch_test_positives=0

                    res_classificator_model.eval()
                    res_encoder_model.eval()

                    for batch in data_loader_test:

                        img_batch = batch['image'].to(args.device)

                        patch_batch = get_patch_tensor_from_image_batch(img_batch)
                        patches_encoded = res_encoder_model.forward(patch_batch)

                        patches_encoded = patches_encoded.view(img_batch.shape[0], 7,7,-1)
                        #patches_encoded = patches_encoded.permute(0,3,1,2)
                        patches_encoded = patches_encoded.permute(0,3,1,2).contiguous()

                        classes = batch['cls'].to(args.device)

                        y_one_hot = torch.zeros(img_batch.shape[0], args.num_classes).to(args.device)
                        y_one_hot = y_one_hot.scatter_(1, classes.unsqueeze(dim=1), 1)

                        labels = batch['class_name']

                        pred = res_classificator_model.forward(patches_encoded)
                        #batch loss
                        loss = torch.sum(-y_one_hot * torch.log(pred))
                        epoch_test_losses.append(loss.detach().to('cpu').numpy())
                        epoch_test_loss += loss.detach().to('cpu').numpy()
                        batch_test_loss+=loss.detach().to('cpu').numpy()
                        batch_test_positives+=float(torch.sum(pred.argmax(dim=1) == classes))
                        

                        epoch_test_true_positives += torch.sum(pred.argmax(dim=1) == classes)

                        #draw confusion matrix
                        #if epoch ==EPOCHS-1 and (t in {len(enumerate(data_loader_train))-1,len(enumerate(data_loader_train))-2}):
                        preds=pred.argmax(dim=1)
                        all_preds = torch.cat((all_preds.float(), preds.float()), dim=0)
                        all_inds=torch.cat((all_inds.float(), classes.float()), dim=0)
                    batch_test_acc=batch_test_positives/float(NUM_TEST_SAMPLES)
                    test_stats = dict(
                        #iteration=t*(epoch+1)
                        batch_test_loss = batch_test_loss,
                        batch_test_acc= batch_test_acc
                    )

                    print("Writing dict {} into file {}".format(test_stats,test_csv_path))
                    write_csv_stats(test_csv_path, test_stats)
                    if best_epoch_test_loss > epoch_test_loss:

                        best_epoch_test_loss = batch_test_loss
                        torch.save(res_encoder_model.state_dict(), os.path.join(models_store_path, "best_res_ecoder_weights.pt"))
                        torch.save(res_classificator_model.state_dict(), os.path.join(models_store_path, "best_res_classificator_weights.pt"))

           


        epoch_train_accuracy = float(epoch_train_true_positives) / float(NUM_TRAIN_SAMPLES)

        print(f"Training loss of epoch {epoch} is {epoch_train_loss}")
        print(f"Accuracy of epoch {epoch} is {epoch_train_accuracy}")

        epoch_test_accuracy = float(epoch_test_true_positives) / float(NUM_TEST_SAMPLES)

        print(f"Test loss of epoch {epoch} is {epoch_test_loss}")
        print(f"Test accuracy of epoch {epoch} is {epoch_test_accuracy}")

        #draw confusion matrix
        cm_test = confusion_matrix(all_inds.cpu(), all_preds.cpu())
        plot_confusion_matrix(cm_test,CLASS_TUPLE,os.path.join(models_store_path, "test_confusion_matrix.jpg"))


        res_classificator_model.train()
        res_encoder_model.train()


        torch.save(res_encoder_model.state_dict(), os.path.join(models_store_path, "last_res_ecoder_weights.pt"))
        torch.save(res_classificator_model.state_dict(), os.path.join(models_store_path, "last_res_classificator_weights.pt"))

        if best_epoch_test_loss > epoch_test_loss:

            best_epoch_test_loss = epoch_test_loss
            torch.save(res_encoder_model.state_dict(), os.path.join(models_store_path, "best_res_ecoder_weights.pt"))
            torch.save(res_classificator_model.state_dict(), os.path.join(models_store_path, "best_res_classificator_weights.pt"))


        stats = dict(
            epoch = epoch,
            train_acc = epoch_train_accuracy,
            train_loss = epoch_train_loss,
            test_acc = epoch_test_accuracy,
            test_loss = epoch_test_loss
        )

        print("Writing dict {} into file {}".format(stats,stats_csv_path))
        write_csv_stats(stats_csv_path, stats)

