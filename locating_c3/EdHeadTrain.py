# consensually sloten and adapted from https://github.com/rrr-uom-projects/3DSegmentationNetwork/blob/master/headHunter
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import EdHeadUtils as utils
import time

#####################################################################################################
##################################### headHunter trainers ###########################################
#####################################################################################################
class headHunter_trainer:
    def __init__(self, model, optimizer, lr_scheduler, device, train_loader, val_loader, logger, checkpoint_dir, max_num_epochs=100,
                num_iterations=1, num_epoch=0, patience=10, iters_to_accumulate=4, best_eval_score=None, eval_score_higher_is_better=False):
        self.logger = logger
        self.logger.info(model)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.eval_score_higher_is_better = eval_score_higher_is_better
        # initialize the best_eval_score
        if not best_eval_score:
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')
        else:
            self.best_eval_score = best_eval_score
        self.patience = patience
        self.epochs_since_improvement = 0
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        self.fig_dir = os.path.join(checkpoint_dir, 'figs')
        try:
            os.mkdir(self.fig_dir)
        except OSError:
            pass
        self.num_iterations = num_iterations
        self.iters_to_accumulate = iters_to_accumulate
        self.num_epoch = num_epoch
        self.epsilon = 1e-6
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self):
        self._save_init_state()
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            t = time.time()
            should_terminate = self.train(self.train_loader)
            print("Epoch trained in " + str(int(time.time()-t)) + " seconds.")
            if should_terminate:
                print("Hit termination condition...")
                break
            self.num_epoch += 1
        self.writer.close()
        return self.num_iterations, self.best_eval_score

    def train(self, train_loader):
        """Trains the model for 1 epoch.
        Args:
            train_loader (torch.utils.data.DataLoader): training data loader
        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = utils.RunningAverage()
        improved = False        # for early stopping
        self.model.train()      # set the model in training mode
        for batch_idx, sample in enumerate(train_loader):
            self.logger.info(f'Training iteration {self.num_iterations}. Batch {batch_idx + 1}. Epoch [{self.num_epoch + 1}/{self.max_num_epochs}]')
            #bodyMask = sample['bodyMask'].type(torch.HalfTensor) 
            ct_im = sample[0].type(torch.FloatTensor)
            #target = sample['target'].numpy()
            h_target = sample[1].type(torch.FloatTensor) 
            # send tensors to GPU
            ct_im = ct_im.to(self.device)
            h_target = h_target.to(self.device)
            
            # forward
            output, loss = self._forward_pass(ct_im, h_target)
            train_losses.update(loss.item(), self._batch_size(ct_im))
            
            # compute gradients and update parameters
            # simulate larger batch sizes using gradient accumulation
            loss = loss/self.iters_to_accumulate

            # Native AMP training step
            self.scaler.scale(loss).backward()
            
            # Every iters_to_accumulate, call step() and reset gradients:
            if self.num_iterations%self.iters_to_accumulate == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                # log stats
                self.logger.info(f'Training stats. Loss: {train_losses.avg}')
                self._log_stats('train', train_losses.avg)
            
            self.num_iterations += 1

        # evaluate on validation set
        self.model.eval()
        eval_score = self.validate()

        # adjust learning rate if necessary
        self.scheduler.step(eval_score)

        # log current learning rate in tensorboard
        self._log_lr()

        # remember best validation metric
        is_best = self._is_best_eval_score(eval_score)
        if(is_best):
            improved = True
        
        # save checkpoint
        self._save_checkpoint(is_best)

        # implement early stopping here
        if not improved:
            self.epochs_since_improvement += 1
        if(self.epochs_since_improvement > self.patience):  # Model has not improved for certain number of epochs
            self.logger.info(
                    f'Model not improved for {self.patience} epochs. Finishing training...')
            return True
        return False    # Continue training...
        

    def validate(self):
        self.logger.info('Validating...')
        val_losses = utils.RunningAverage()
        with torch.no_grad():
            which_to_show = np.random.randint(0, self.val_loader.batch_size)
            for batch_idx, sample in enumerate(self.val_loader):
                self.logger.info(f'Validation iteration {batch_idx + 1}')
                ct_im = sample[0].type(torch.FloatTensor) 
                #target = sample['target'].numpy()
                h_target = sample[1].type(torch.FloatTensor)  
                
                # send tensors to GPU
                ct_im = ct_im.to(self.device)
                h_target = h_target.to(self.device)
                
                output, loss = self._forward_pass(ct_im, h_target)
                val_losses.update(loss.item(), self._batch_size(ct_im))
                
                if (batch_idx == 0) and ((self.num_epoch < 100) or (self.num_epoch < 500 and not self.num_epoch%10) or (not self.num_epoch%100)):
                    # plot im
                    #target = target[which_to_show]
                    h_target = h_target.cpu().numpy()[which_to_show]
                    output = output.cpu().numpy()[which_to_show]
                    print(f'target: {h_target}')
                    #pred = np.unravel_index(output[0].argmax(), output[0].shape)
                    #print(f'prediction: {pred}, map value: {output[0,pred[0],pred[1],pred[2]]}')
                    '''
                    # CoM of Parotids and Brainstem plots
                    # axial plot
                    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
                    ax_slice = ct_im.cpu().numpy()[which_to_show, 2, int(target[0])]              # <-- batch_num, contrast_channel, ax_slice
                    ax0.imshow(ax_slice, aspect=1.0, cmap='Greys_r')
                    ax_slice = h_target[0, int(target[0])]
                    ax1.imshow(ax_slice, aspect=1.0, cmap='nipy_spectral', vmin=0, vmax=max(self.epsilon,np.max(h_target)))
                    ax_slice = output[0, int(target[0])]
                    ax2.imshow(ax_slice, aspect=1.0, cmap='nipy_spectral', vmin=0, vmax=max(self.epsilon,np.max(output)))
                    self.writer.add_figure(tag='Val_pred_ax', figure=fig, global_step=self.num_epoch)
                    fig.savefig(os.path.join(self.fig_dir, 'Val_pred_ax_'+str(self.num_epoch)+'.png'))
                    # sagittal plot
                    fig2, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(10, 3), tight_layout=True)
                    sag_slice = ct_im.cpu().numpy()[which_to_show,2,:,:,int(target[2])]
                    ax3.imshow(sag_slice, aspect=2.0, cmap='Greys_r')
                    sag_slice = h_target[0, :, :, int(target[2])]
                    ax4.imshow(sag_slice, aspect=2.0, cmap='nipy_spectral', vmin=0, vmax=max(self.epsilon,np.max(h_target)))
                    sag_slice = output[0, :, :, int(target[2])]
                    ax5.imshow(sag_slice, aspect=2.0, cmap='nipy_spectral', vmin=0, vmax=max(self.epsilon,np.max(output)))
                    self.writer.add_figure(tag='Val_pred_sag', figure=fig2, global_step=self.num_epoch)
                    #fig2.savefig(os.path.join(self.fig_dir, 'Val_pred_sag_'+str(self.num_epoch)+'.png'))
                    '''                    
                    '''
                    # Parotid figures
                    # axial plot - lpar
                    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
                    ax_slice = ct_im.cpu().numpy()[which_to_show, 2, int(target[0,0])]              # <-- batch_num, contrast_channel, ax_slice (loc_idx, axial_idx)
                    ax0.imshow(ax_slice, aspect=1.0, cmap='Greys_r')
                    ax_slice = h_target[0, int(target[0,0])]
                    ax1.imshow(ax_slice, aspect=1.0, cmap='nipy_spectral', vmin=0, vmax=max(self.epsilon,np.max(h_target)))
                    ax_slice = output[0, int(target[0,0])]
                    ax2.imshow(ax_slice, aspect=1.0, cmap='nipy_spectral', vmin=0, vmax=max(self.epsilon,np.max(output)))
                    self.writer.add_figure(tag='Val_lpar_pred_ax', figure=fig, global_step=self.num_epoch)
                    fig.savefig(os.path.join(self.fig_dir, 'Val_lpar_pred_ax_'+str(self.num_epoch)+'.png'))
                    
                    # axial plot - rpar
                    fig2, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
                    ax_slice = ct_im.cpu().numpy()[which_to_show, 2, int(target[1,0])]              # <-- batch_num, contrast_channel, ax_slice (loc_idx, axial_idx)
                    ax3.imshow(ax_slice, aspect=1.0, cmap='Greys_r')
                    ax_slice = h_target[1, int(target[1,0])]
                    ax4.imshow(ax_slice, aspect=1.0, cmap='nipy_spectral', vmin=0, vmax=max(self.epsilon,np.max(h_target)))
                    ax_slice = output[1, int(target[1,0])]
                    ax5.imshow(ax_slice, aspect=1.0, cmap='nipy_spectral', vmin=0, vmax=max(self.epsilon,np.max(output)))
                    self.writer.add_figure(tag='Val_rpar_pred_ax', figure=fig2, global_step=self.num_epoch)
                    fig2.savefig(os.path.join(self.fig_dir, 'Val_rpar_pred_ax_'+str(self.num_epoch)+'.png'))
                    '''
                    # Spine targetted plots
                    # sagittal plots
                    # for target_idx in range(target.shape[0]):
                    #     fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 3), tight_layout=True)
                    #     sag_slice = ct_im.cpu().numpy()[which_to_show,0,:,:,int(target[target_idx,2])]
                    #     ax0.imshow(sag_slice, aspect=2.0, cmap='Greys_r', vmin=ct_im.cpu().numpy().min(), vmax=ct_im.cpu().numpy().max())
                    #     sag_slice = h_target[target_idx, :, :, int(target[target_idx,2])]
                    #     ax1.imshow(sag_slice, aspect=2.0, cmap='nipy_spectral', vmin=0, vmax=max(self.epsilon,np.max(h_target)))
                    #     sag_slice = output[target_idx, :, :, int(target[target_idx,2])]
                    #     ax2.imshow(sag_slice, aspect=2.0, cmap='nipy_spectral', vmin=0, vmax=max(self.epsilon,np.max(output)))
                    #     self.writer.add_figure(tag='Val_pred_sag_'+str(target_idx), figure=fig, global_step=self.num_epoch)
                    
            self._log_stats('val', val_losses.avg)
            self.logger.info(f'Validation finished. Loss: {val_losses.avg}')
            return val_losses.avg

    def _forward_pass(self, ct_im, h_target):
        with torch.cuda.amp.autocast():
            # forward pass
            output = self.model(ct_im)
            # MSE loss contribution - unchanged for >1 targets
            loss = torch.nn.MSELoss()(output, h_target)
            # L1 loss contribution
            output = output.cpu()
            if (output.shape[1] == 1):
                # single target case
                #target_vox = torch.tensor(np.round(target)).type(torch.FloatTensor)
                pred_vox = torch.tensor([np.unravel_index(torch.argmax(output[i, 0]), output.size()[2:]) for i in range(output.size(0))]).type(torch.FloatTensor)
            #else:

                # multi-target case
                #target_vox = torch.tensor(np.round(target)).type(torch.FloatTensor)
                #pred_vox = torch.zeros(output.shape[:2]+(3,))   # Important to capture batch 
                # for loc_idx in range(output.shape[1]):
                #    single_pred_vox = torch.tensor([np.unravel_index(torch.argmax(output[i, loc_idx]), output.size()[2:]) for i in range(output.size(0))]).type(torch.FloatTensor)
                #    pred_vox[:, loc_idx] = single_pred_vox
            loss += (torch.nn.L1Loss()(pred_vox) * 0.01) # scaling factor for the L1 supplementary term
            return output, loss

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score
        if is_best:
            self.logger.info(f'Saving new best evaluation metric: {eval_score}')
            self._log_new_best(eval_score)
            self.best_eval_score = eval_score
            self.epochs_since_improvement = 0
        return is_best

    def _save_init_state(self):
        state = {'model_state_dict': self.model.state_dict()}
        init_state_path = os.path.join(self.checkpoint_dir, 'initial_state.pytorch')
        self.logger.info(f"Saving initial state to '{init_state_path}'")
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        torch.save(state, init_state_path)

    def _save_checkpoint(self, is_best):
        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            #'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_new_best(self, eval_score):
        self.writer.add_scalar('best_val_loss', eval_score, self.num_iterations)

    def _log_stats(self, phase, loss_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
        }
        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            #self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations) #not sure what this is 

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)