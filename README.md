# Stochastic Value Gradients with Priors
A PyTorch implemenation of the SVG(0) algorithm with extension of KL regularized, and behavior priors.

## Setup
Install the packages in `requirements.txt`. 

## Usage
Run experiments by using the following example command:

```bash
python main.py --name pendulum_svg0_kl_prior --a svg0_prior -c configs/svg0_kl_prior.yml
```

Arguments
```
--name: Name of the experiment
-d, --debug: Enable debug mode so model parameters are not stored.
-a, --alg: Algorithm selection, choices: {svg0, svg0_kl_prior}
-c, --config: Location of the config files, see /configs
```

## Algorithms

- [x] SVG(0)
- [x] SVG(0) with KL regularized prior
- [ ] SVG(∞)
- [ ] ~~SVG(0) with behavior priors~~


## Results

TODO


## References
- Heess, N., Wayne, G., Silver, D., Lillicrap, T. P., Tassa, Y., & Erez, T. (2015). Learning Continuous Control Policies by Stochastic Value Gradients. CoRR, abs/1510.09142. Retrieved from http://arxiv.org/abs/1510.09142
- Galashov, A., Jayakumar, S. M., Hasenclever, L., Tirumala, D., Schwarz, J., Desjardins, G., … Heess, N. (2019). Information asymmetry in KL-regularized RL. CoRR, abs/1905.01240. Retrieved from http://arxiv.org/abs/1905.01240
- Tirumala, D., Galashov, A., Noh, H., Hasenclever, L., Pascanu, R., Schwarz, J., … Heess, N. (2022). Behavior Priors for Efficient Reinforcement Learning. Journal of Machine Learning Research, 23(221), 1–68. Retrieved from http://jmlr.org/papers/v23/20-1038.html


## Acknowledgements
I would like to thank the authors of the following repository in particular. They were great help to me for understanding the implementation details of SVG0.
- https://github.com/philipjball/OffCon3