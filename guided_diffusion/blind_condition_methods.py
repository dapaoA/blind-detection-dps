from typing import Dict
import torch

from guided_diffusion.measurements import BlindBlurOperator, TurbulenceOperator, InpaintingOperator
from guided_diffusion.condition_methods import ConditioningMethod, register_conditioning_method

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


class BlindConditioningMethod(ConditioningMethod):
    def __init__(self, operator, noiser=None, **kwargs):
        '''
        Handle multiple score models.
        Yet, support only gaussian noise measurement.
        '''
        assert isinstance(operator, BlindBlurOperator) or isinstance(operator, TurbulenceOperator)
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, kernel, noisy_measuerment, **kwargs):
        return self.operator.project(data=data, kernel=kernel, measurement=noisy_measuerment, **kwargs)

    def grad_and_value(self, 
                       x_prev: Dict[str, torch.Tensor], 
                       x_0_hat: Dict[str, torch.Tensor], 
                       measurement: torch.Tensor,
                       **kwargs):
        reg_norms = []
        if self.noiser.__name__ == 'gaussian':  # why none?
            
            assert sorted(x_prev.keys()) == sorted(x_0_hat.keys()), \
                "Keys of x_prev and x_0_hat should be identical."

            keys = sorted(x_prev.keys())
            x_prev_values = [x[1] for x in sorted(x_prev.items())] 
            x_0_hat_values = [x[1] for x in sorted(x_0_hat.items())]
            
            difference = measurement - self.operator.forward(*x_0_hat_values)
            norm = torch.linalg.norm(difference)

            reg_info = kwargs.get('regularization', None)
            if reg_info is not None:
                for reg_target in reg_info:
                    assert reg_target in keys, \
                        f"Regularization target {reg_target} does not exist in x_0_hat."

                    reg_ord, reg_scale = reg_info[reg_target]
                    if reg_scale != 0.0:  # if got scale 0, skip calculating.
                        reg_norms.append(reg_scale * torch.linalg.norm(x_0_hat[reg_target].view(-1), ord=reg_ord))
            if reg_norms:
                norm = norm + sum(reg_norms)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev_values)
        elif self.noiser.__name__ == "directe":
            assert sorted(x_prev.keys()) == sorted(x_0_hat.keys()), \
                "Keys of x_prev and x_0_hat should be identical."

            keys = sorted(x_prev.keys())
            x_prev_values = [x[1] for x in sorted(x_prev.items())] 
            x_0_hat_values = [x[1] for x in sorted(x_0_hat.items())]
            
            difference = measurement - self.operator.forward(*x_0_hat_values)
            norm = torch.linalg.norm(difference)

            reg_info = kwargs.get('regularization', None)
            if reg_info is not None:
                for reg_target in reg_info:
                    assert reg_target in keys, \
                        f"Regularization target {reg_target} does not exist in x_0_hat."

                    reg_ord, reg_scale = reg_info[reg_target]
                    if reg_scale != 0.0:  # if got scale 0, skip calculating.
                        reg_norms.append(reg_scale * torch.linalg.norm(x_0_hat[reg_target].view(-1), ord=reg_ord))
            if reg_norms:
                norm = norm + sum(reg_norms)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev_values)
        else:
            raise NotImplementedError
        
        return dict(zip(keys, norm_grad)), norm


class PartialBlindConditioningMethod(ConditioningMethod):
    def __init__(self, operator, noiser=None, **kwargs):
        '''
        Handle multiple score models.
        Yet, support only gaussian noise measurement.
        '''
        assert isinstance(operator, BlindBlurOperator) or isinstance(operator, TurbulenceOperator) or isinstance(operator, InpaintingOperator)
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, kernel, noisy_measuerment, **kwargs):
        return self.operator.project(data=data, kernel=kernel, measurement=noisy_measuerment, **kwargs)

    def grad_and_value(self, 
                       x_prev: Dict[str, torch.Tensor], 
                       x_0_hat: Dict[str, torch.Tensor], 
                       measurement: torch.Tensor,
                       **kwargs):
        reg_norms = []
        if self.noiser.__name__ == 'gaussian':  # why none?
            
            assert sorted(x_prev.keys()) == sorted(x_0_hat.keys()), \
                "Keys of x_prev and x_0_hat should be identical."

            keys = sorted(x_prev.keys())
            x_prev_values = [x[1] for x in sorted(x_prev.items())] 
            x_0_hat_values = [x[1] for x in sorted(x_0_hat.items())]
            
            difference = measurement - self.operator.forward(*x_0_hat_values)
            norm = torch.linalg.norm(difference)

            reg_info = kwargs.get('regularization', None)
            if reg_info is not None:
                for reg_target in reg_info:
                    assert reg_target in keys, \
                        f"Regularization target {reg_target} does not exist in x_0_hat."

                    reg_ord, reg_scale = reg_info[reg_target]
                    if reg_scale != 0.0:  # if got scale 0, skip calculating.
                        reg_norms.append(reg_scale * torch.linalg.norm(x_0_hat[reg_target].view(-1), ord=reg_ord))
            if reg_norms:
                norm = norm + sum(reg_norms)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev_values)
        elif self.noiser.__name__ == "directe":
            assert sorted(x_prev.keys()) == sorted(x_0_hat.keys()), \
                "Keys of x_prev and x_0_hat should be identical."

            keys = sorted(x_prev.keys())
            x_prev_values = [x[1] for x in sorted(x_prev.items())] 
            x_0_hat_values = [x[1] for x in sorted(x_0_hat.items())]
            
            difference = measurement - self.operator.forward(*x_0_hat_values)
            norm = torch.linalg.norm(difference)

            reg_info = kwargs.get('regularization', None)
            if reg_info is not None:
                for reg_target in reg_info:
                    assert reg_target in keys, \
                        f"Regularization target {reg_target} does not exist in x_0_hat."

                    reg_ord, reg_scale = reg_info[reg_target]
                    if reg_scale != 0.0:  # if got scale 0, skip calculating.
                        reg_norms.append(reg_scale * torch.linalg.norm(x_0_hat[reg_target].view(-1), ord=reg_ord))
            if reg_norms:
                norm = norm + sum(reg_norms)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev_values)
        else:
            raise NotImplementedError
        
        return dict(zip(keys, norm_grad)), norm

    def grad_and_value_for_mask(self, 
                       x_prev: torch.Tensor, 
                       x_0_hat: torch.Tensor, 
                       measurement: torch.Tensor,
                       **kwargs):
        reg_norms = []
        if self.noiser.__name__ == 'gaussian':  # why none?
            x_prev = x_prev.requires_grad_()
            difference = measurement - self.operator.forward(x_0_hat, mask=x_0_hat)
            norm = torch.linalg.norm(difference)

            reg_info = kwargs.get('regularization', {})
            if reg_info:
                reg_ord = reg_info.get('ord', 2)
                reg_scale = reg_info.get('scale', 0.0)
                if reg_scale != 0.0:  # if got scale 0, skip calculating.
                    reg_norms.append(reg_scale * torch.linalg.norm(x_0_hat.view(-1), ord=reg_ord))
            if reg_norms:
                norm = norm + sum(reg_norms)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=[x_prev])[0]
        elif self.noiser.__name__ == "directe":
            x_prev = x_prev.requires_grad_()
            difference = measurement - self.operator.forward(x_0_hat)
            norm = torch.linalg.norm(difference)

            reg_info = kwargs.get('regularization', {})
            if reg_info:
                reg_ord = reg_info.get('ord', 2)
                reg_scale = reg_info.get('scale', 0.0)
                if reg_scale != 0.0:  # if got scale 0, skip calculating.
                    reg_norms.append(reg_scale * torch.linalg.norm(x_0_hat.view(-1), ord=reg_ord))
            if reg_norms:
                norm = norm + sum(reg_norms)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=[x_prev])[0]
        else:
            raise NotImplementedError
        
        return norm_grad, norm

@register_conditioning_method(name='ps')
class PosteriorSampling(BlindConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        assert kwargs.get('scale') is not None
        self.scale = kwargs.get('scale')

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev, x_0_hat, measurement, **kwargs)

        scale = kwargs.get('scale')
        if scale is None:
            scale = self.scale
         
        keys = sorted(x_prev.keys())
        for k in keys:
            x_t.update({k: x_t[k] - scale[k]*norm_grad[k]})            
        
        return x_t, norm
    

@register_conditioning_method(name='emdps')
class EMDPosteriorSampling(PartialBlindConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        assert kwargs.get('scale') is not None
        self.scale = kwargs.get('scale')
        self.iterations = kwargs.get('iterations', {'img': 1, 'kernel': 5})

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        scale = kwargs.get('scale')
        iterations = kwargs.get('iterations')
        if scale is None:
            scale = self.scale
        if iterations is None:
            iterations = self.iterations
        keys = sorted(x_prev.keys())
        norm = None
        # Handle img key first
        if 'img' in keys:
            for _ in range(iterations['img']):
                norm_grad, norm = self.grad_and_value_for_mask(x_prev['img'], x_0_hat['img'], measurement, **kwargs)
                x_t['img'] = x_t['img'] - scale['img']*norm_grad/torch.linalg.norm(norm_grad)
        
        # Handle other keys (e.g. circle parameters)
        for k in keys:
            if k != 'img':
                for _ in range(iterations[k]):
                    x_0_hat[k] = x_0_hat[k].detach().requires_grad_()
                    
                    H, W = x_t['img'].shape[-2:]
                    y, x = torch.meshgrid(torch.arange(H, device=x_0_hat[k].device), 
                                        torch.arange(W, device=x_0_hat[k].device))
                    
                    batch_size = x_0_hat[k].shape[0]
                    x_coords = x_0_hat[k][:, 1]
                    y_coords = x_0_hat[k][:, 2]
                    radii = x_0_hat[k][:, 0]
                    
                    x_grid = x[None, :, :].expand(batch_size, -1, -1)
                    y_grid = y[None, :, :].expand(batch_size, -1, -1)
                    
                    # 计算平滑的距离场
                    dist = torch.sqrt((x_grid - x_coords.unsqueeze(-1).unsqueeze(-1))**2 + 
                                    (y_grid - y_coords.unsqueeze(-1).unsqueeze(-1))**2 + 1e-8)
                    
                    # 使用 softmax 创建平滑的掩码
                    # temperature 控制平滑程度，越小越接近原始的阶跃函数
                    temperature = 1
                    normalized_dist = (dist - radii.unsqueeze(-1).unsqueeze(-1)) / temperature
                    circle_mask = torch.ones((batch_size, 1, H, W), device=x_0_hat[k].device)
                    circle_mask[:, 0] = torch.sigmoid(normalized_dist)
                    
                    # 计算损失
                    difference = measurement - circle_mask
                    mask_loss = torch.linalg.norm(difference)
                    radius_penalty = 0.01 * torch.sum(radii**2)
                    
                    # 打印两部分损失
                    # print(f"Mask loss: {mask_loss.item():.4f}, Radius penalty: {radius_penalty.item():.4f}")
                    
                    # 总损失
                    norm = mask_loss + radius_penalty
                    
                    # 计算梯度并更新
                    norm_grad = torch.autograd.grad(outputs=norm, inputs=[x_0_hat[k]])[0]
                    x_0_hat[k] = x_0_hat[k] - 0 * scale[k]*norm_grad/torch.linalg.norm(norm_grad)
                    x_0_hat[k] = x_0_hat[k].detach()
                    # print(f"x_0_hat[{k}]", x_0_hat[k])
                    
                    x_t[k] = x_0_hat[k]

        return x_t, norm