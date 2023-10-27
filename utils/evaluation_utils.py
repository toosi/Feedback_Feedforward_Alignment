import torch
import time
import numpy as np
import torch.nn.functional as F


from utils import helper_functions

def validate(val_loader, modelF, modelB, criterione, criteriond, args, itr, sigma2):

    
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    corr = AverageMeter('corr', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    m1, m2 = top1, corr

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, m1, m2],
        prefix='Test %s: '%args.method)

    if args.gpu is not None:
        
        onehot = torch.FloatTensor(args.batch_size, args.n_classes).cuda(args.gpu, non_blocking=True)
    else:
        onehot = torch.FloatTensor(args.batch_size, args.n_classes).cuda()


    # switch to evaluate mode
    modelF.eval()
    modelB.eval()

    not_saved = True
    not_saved_itr = True

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            else:
                images = images.cuda()
                target = target.cuda()
            
            onehot.zero_()
            onehot.scatter_(1, target.view(target.shape[0], 1), 1)

            if 'MNIST' in args.dataset  and args.arche[0:2]!='FC':
                images= images.expand(-1, 1, -1, -1) # images.expand(-1, 3, -1, -1)
            
            # ----- encoder ---------------------
            images_noisy = images + torch.empty_like(images).normal_(mean=0, std=np.sqrt(sigma2)).cuda()
            
            if (args.eval_save_sample_images) and not_saved:
                not_saved = False
                helper_functions.generate_sample_images(images, target, title='original', param_dict={}, args=args)
                helper_functions.generate_sample_images(images_noisy, target, title='noisy inputs', param_dict={'sigma2':sigma2}, args=args)

            # compute output
            latents, output = modelF(images_noisy)
            # ----- decoder ------------------ 
            _,recons_before_interpolation = modelB(latents.detach()) 
            recons = F.interpolate(recons_before_interpolation, size=images.shape[-1])
            
            
            gener = recons
            reference = images

        

            for _ in range(itr):
                # compute output
                latents, output = modelF(gener.detach())

                # ----- decoder ------------------ 
                _, recons_before_interpolation = modelB(latents.detach()) 
                recons = F.interpolate(recons_before_interpolation, size=images.shape[-1])

                gener = recons
                reference = images
            
            
            if (args.eval_save_sample_images) and not_saved_itr:
                not_saved_itr = False
                helper_functions.generate_sample_images(gener, target, title='gener by '+args.method, param_dict={'sigma2':sigma2, 'itr':itr}, args=args)

            
            # measure accuracy and record loss
            losse = criterione(output, target) 
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0].item(), images.size(0))

            # measure correlation and record loss
            reference = F.interpolate(reference, size=gener.shape[-1])
            lossd = criteriond(gener, reference) 

            pcorr = correlation(gener, reference)
            losses.update(lossd.item(), images.size(0))
            corr.update(pcorr, images.size(0))
                
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)


        print('Test avg {method} sigma2 {sigma2} itr: {itr} * lossd {losses.avg:.3f}'
            .format(method=args.method, sigma2=sigma2, itr=itr,losses=losses), flush=True)

        # TODO: this should also be done with the ProgressMeter
        print('Test avg  {method} sigma2 {sigma2} itr: {itr} * Acc@1 {top1.avg:.3f}'
            .format(method=args.method, sigma2=sigma2, itr=itr, top1=top1), flush=True)
              

    return modelF, modelB, [top1.avg, corr.avg, losses.avg]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def correlation(output, images):
    """Computes the correlation between reconstruction and the original images"""
    x = output.contiguous().view(-1)
    y = images.contiguous().view(-1) 

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    pearson_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return pearson_corr.item()
    

