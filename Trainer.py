# Trainer Class

import os
import torch
from ConfusionMatrix import plot_confusion_matrix

class Trainer:
  def __init__(self, log_dir, n_epochs=3, device="cpu"):
    self.model = None
    self.optimiser = None
    self.valid = None
    self.data = None
    self.device = device
    self.max_epochs = n_epochs
    self.avg_train_loss = []
    self.avg_valid_loss = []
    self.training_accuracy = []
    self.validation_accuracy = []
    self.log_dir = log_dir

  @staticmethod
  def exp_lr(lr):
    return lr * 0.95

  # The fitting step
  def fit(self, model, data, valid, resume=False, completed_epochs=None, use_lr_scheduler=True):
    self.data = data
    self.valid = valid

    # configure the optimizer
    self.optimiser = model.configure_optimiser()
    self.model = model.to(self.device)
    self.model.train()

    if resume and completed_epochs is None:
      print("If resuming, must pass the number of completed epochs.")
      return

    for epoch in range(self.max_epochs):
      print("Epoch: ", epoch + completed_epochs if resume else epoch)
      print("Training:")
      self.model.train()
      self.avg_train_loss.append(self.fit_epoch())
      if use_lr_scheduler:
        self.model.lr = self.exp_lr(self.model.lr)
      self.evaluate(self.model, self.data, train=True, plot_cm=False)

      self.model.eval()
      print("Validating:")
      self.avg_valid_loss.append(self.validate_epoch())
      self.evaluate(self.model, self.valid, train=False, plot_cm=True)

      if epoch % 5 == 0 or epoch == self.max_epochs - 1:
         self.model.save(save_dir=self.log_dir, trained_epochs=epoch)
         self.save(save_dir=self.log_dir, trained_epochs=epoch)

    print("Training process has finished")

  def fit_epoch(self):
    current_loss = 0.0
    overall_loss = 0.0

    for i, data in enumerate(self.data):
      # Get input and its corresponding groundtruth output
      inputs, target = data
      inputs, target = inputs.to(self.device), target.to(self.device)

      self.optimiser.zero_grad()

      # get output from the model, given the inputs
      outputs = self.model(inputs)

      # get loss for the predicted output
      loss = self.model.loss(y_hat=outputs, y=target)

      # get gradients w.r.t the parameters of the model
      loss.backward()

      # update the parameters (perform optimization)
      self.optimiser.step()

      current_loss += loss.item()
      overall_loss += loss.item()
      if i % 10 == 9:
          print('Loss after mini-batch %5d: %.3f' %
                (i + 1, current_loss / 10))
          current_loss = 0.0
    return overall_loss / len(self.data.dataset)

  @torch.no_grad()
  def validate_epoch(self):
    current_loss = 0.0
    overall_loss = 0.0

    for i, data in enumerate(self.valid):
      inputs, target = data
      inputs, target = inputs.to(self.device), target.to(self.device)

      outputs = self.model(inputs)

      current_loss += self.model.loss(outputs, target).item()
      overall_loss += current_loss
      if i % 10 == 9:
          print('Loss after mini-batch %5d: %.3f' %
                (i + 1, current_loss / 10))
          current_loss = 0.0
    return overall_loss / len(self.valid.dataset)

  @torch.no_grad()
  def evaluate(self, model, data, train=False, plot_cm=False):
    correct = 0
    total = 0
    predictions = []
    truths = []

    for images, labels in data:
      images, labels = images.to(self.device), labels.to(self.device)
      outputs = model(images)
      est_label = torch.max(outputs, 1).indices
      correct += sum(est_label == labels)
      total += labels.size(0)
      for label in est_label.cpu():
          predictions.append(label)
      for label in labels.cpu():
          truths.append(label)

    print("Training Accuracy: " if train else "Validation Accuracy: ", float((correct / total) * 100), "%")
    self.training_accuracy.append(float((correct / total))) if train else self.validation_accuracy.append(float((correct / total)))
    print("Correct: ", correct, "Total: ", total)

    if plot_cm:
        plot_confusion_matrix(save_dir=self.log_dir, est=predictions, gnd_truth=truths, labels=self.data.dataset.classes)

  def save(self, save_dir, trained_epochs=0):
    save_path = (save_dir + f"/Epoch_{int(trained_epochs)}_LossAccuracy.tar")
    torch.save(
        dict(training_loss=self.avg_train_loss,
             validation_loss=self.avg_valid_loss,
             training_accuracy=self.training_accuracy,
             validation_accuracy=self.validation_accuracy),
        save_path)
    print(f"MLP Training Loss saved to {save_path} at Epoch {trained_epochs}")

  def load(self, load_dir):
    checkpoint = torch.load(load_dir, weights_only=True)
    self.avg_train_loss = checkpoint['training_loss']
    self.avg_valid_loss = checkpoint['validation_loss']
    self.training_accuracy = checkpoint['training_accuracy']
    self.validation_accuracy = checkpoint['validation_accuracy']