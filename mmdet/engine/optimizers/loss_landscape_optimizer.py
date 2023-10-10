import torch
from mmdet.registry import OPTIMIZERS

@OPTIMIZERS.register_module()
class LandScape(torch.optim.Optimizer):
    def __init__(self, params, x_min=-0.1, x_max=0.1, x_num=51, y_min=-0.1, y_max=0.1, y_num=51, **kwargs):
        defaults = dict(**kwargs)
        super(LandScape, self).__init__(params, defaults)

        self.x_min, self.x_max, self.x_num = x_min, x_max, x_num
        self.y_min, self.y_max, self.y_num = y_min, y_max, y_num

        self._coords = torch.meshgrid(torch.linspace(self.x_min, self.x_max, steps=self.x_num),
                                      torch.linspace(self.y_min, self.y_max, steps=self.y_num))  # 51*51
        self._x_directions = self.noise_weights() # [[197 all params][197]]
        self._y_directions = self.noise_weights()
        self._current_step_x = 0
        self._current_step_y = 0

    def noise_weights(self):
        directions = []
        for group in self.param_groups:
            random_weights = [torch.randn(p.size()).to(p) for p in group["params"]]
            for d, w in zip(random_weights, group["params"]):
                d.mul_(w.norm() / (d.norm() + 1e-10))
            directions.append(random_weights)

        return directions

    @torch.no_grad()
    def step(self):
        assert self._current_step_x < self.x_num
        assert self._current_step_y < self.y_num

        coord_x = self._coords[0][self._current_step_x][self._current_step_y]
        coord_y = self._coords[1][self._current_step_x][self._current_step_y]
        print('coord_x' + str(coord_x))
        print('coord_y' + str(coord_y))
        for group, x_direction, y_direction in zip(self.param_groups, self._x_directions, self._y_directions):
            for p, e_x, e_y in zip(group["params"], x_direction, y_direction):
                if "e_w" in self.state[p]:
                    p.add_(self.state[p]["e_w"], alpha=-1.0)
                e_w = e_x * coord_x + e_y * coord_y
                self.state[p]["e_w"] = e_w
                p.add_(e_w)

        self._current_step_y += 1
        if self._current_step_y >= self.y_num:
            self._current_step_x += 1
            self._current_step_y -= self.y_num