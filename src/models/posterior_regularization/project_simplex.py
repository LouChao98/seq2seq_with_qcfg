# From https://github.com/smatmo/ProjectionOntoSimplex

import torch


def project_simplex(v, z=1.0, axis=-1):
    """
    Implements the algorithm in Figure 1 of
    John Duchi, Shai Shalev-Shwartz, Yoram Singer, Tushar Chandra,
    "Efficient Projections onto the l1-Ball for Learning in High Dimensions", ICML 2008.
    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    This algorithm project vectors v onto the simplex w >= 0, \sum w_i = z.
    :param v: A torch tensor, will be interpreted as a collection of vectors.
    :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
    :param axis: Indicates the axis of v, which defines the vectors to be projected.
    :return: w: result of the projection
    """

    def _project_simplex_2d(v, z):
        """
        Helper function, assuming that all vectors are arranged in rows of v.
        :param v: NxD torch tensor; Duchi et al. algorithm is applied to each row in vecotrized form
        :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
        :return: w: result of the projection
        """
        with torch.no_grad():
            shape = v.shape
            if shape[1] == 1:
                w = v.clone().detach()
                w[:] = z
                return w

            mu = torch.sort(v, dim=1)[0]
            mu = torch.flip(mu, dims=(1,))
            cum_sum = torch.cumsum(mu, dim=1)
            j = torch.unsqueeze(torch.arange(1, shape[1] + 1, dtype=mu.dtype, device=mu.device), 0)
            rho = torch.sum(mu * j - cum_sum + z > 0.0, dim=1, keepdim=True) - 1
            max_nn = cum_sum[torch.arange(shape[0]), rho[:, 0]]
            theta = (torch.unsqueeze(max_nn, -1) - z) / (rho.type(max_nn.dtype) + 1)
            w = torch.clamp(v - theta, min=0.0)
            return w

    with torch.no_grad():
        shape = v.shape

        if len(shape) == 1:
            return _project_simplex_2d(torch.unsqueeze(v, 0), z)[0, :]
        else:
            axis = axis % len(shape)
            t_shape = tuple(range(axis)) + tuple(range(axis + 1, len(shape))) + (axis,)
            tt_shape = tuple(range(axis)) + (len(shape) - 1,) + tuple(range(axis, len(shape) - 1))
            v_t = v.permute(t_shape)
            v_t_shape = v_t.shape
            v_t_unroll = torch.reshape(v_t, (-1, v_t_shape[-1]))

            w_t = _project_simplex_2d(v_t_unroll, z)

            w_t_reroll = torch.reshape(w_t, v_t_shape)
            return w_t_reroll.permute(tt_shape)


if __name__ == "__main__":

    # violations will be larger for float32
    # precision = torch.float32
    precision = torch.float64
    device = "cuda"

    z = 1.0
    overall_nonneg_violation = 0.0
    overall_l1_violation = 0.0
    overall_suboptimality = 0.0

    print("")
    print("**********************")
    print("TEST l1-PROJECT VECTOR")
    print("**********************")

    for d in [1, 2, 3, 5]:
        print("")
        print("Dimensions {}".format(d))

        nonneg_violation = 0.0
        l1_violation = 0.0
        suboptimality = 0.0

        for rep in range(100):
            x = torch.randn(d, dtype=precision, device=device)
            x_projected = project_simplex(x, z)

            if rep == 0:
                print("x:")
                print(x)
                print("projected:")
                print(x_projected)

            nonneg_violation_cur = -torch.min(torch.clamp(x_projected, max=0.0))
            l1_violation_cur = torch.abs(torch.sum(x_projected) - z)

            D = torch.sum((x_projected - x) ** 2) ** 0.5

            x_perturbed = torch.unsqueeze(x_projected, -1) + 0.01 * torch.randn(
                d, 10000, dtype=precision, device=device
            )
            x_perturbed = torch.clamp(x_perturbed, min=0.0)
            x_perturbed /= torch.sum(x_perturbed, dim=0, keepdim=True)
            x_perturbed *= z

            D_perturbed = torch.sum((torch.unsqueeze(x, -1) - x_perturbed) ** 2, dim=0) ** 0.5
            suboptimality_cur = -torch.min(torch.clamp(D_perturbed - D, max=0.0))

            nonneg_violation = max(nonneg_violation_cur, nonneg_violation)
            l1_violation = max(l1_violation_cur, l1_violation)
            suboptimality = max(suboptimality_cur, suboptimality)
            overall_nonneg_violation = max(nonneg_violation_cur, overall_nonneg_violation)
            overall_l1_violation = max(l1_violation_cur, overall_l1_violation)
            overall_suboptimality = max(suboptimality_cur, overall_suboptimality)

        print(
            "Nonnegativety violation {:0.12f}, l1 violation {:0.12f}, suboptimality {:0.12f}".format(
                nonneg_violation, l1_violation, suboptimality
            )
        )

    print("")
    print("")
    print("")

    print("**********************")
    print("TEST l1-PROJECT TENSOR")
    print("**********************")
    for m in [2, 3, 4]:
        for a in range(m):
            for d in [1, 2, 3, 5]:
                print("")
                print("Modes {}, Axis {}, Dimensions {}".format(m, a, d))

                nonneg_violation = 0.0
                l1_violation = 0.0
                suboptimality = 0.0

                x_shape = (10,) * a + (d,) + (10,) * (m - a - 1)
                x = torch.randn(x_shape, dtype=precision, device=device)

                x_projected = project_simplex(x, z, axis=a)

                nonneg_violation_cur = -torch.min(torch.clamp(x_projected, max=0.0))
                l1_violation_cur = torch.max(torch.abs(torch.sum(x_projected, dim=a) - z))

                D = torch.sum((x_projected - x) ** 2, dim=a) ** 0.5

                x_perturbed = x_projected + 0.01 * torch.randn(x_projected.shape, dtype=precision, device=device)
                x_perturbed = torch.clamp(x_perturbed, min=0.0)
                x_perturbed /= torch.sum(x_perturbed, dim=a, keepdim=True)
                x_perturbed *= z

                D_perturbed = torch.sum((x_perturbed - x) ** 2, dim=a) ** 0.5
                suboptimality_cur = -torch.min(torch.clamp(D_perturbed - D, max=0.0))

                nonneg_violation = max(nonneg_violation_cur, nonneg_violation)
                l1_violation = max(l1_violation_cur, l1_violation)
                suboptimality = max(suboptimality_cur, suboptimality)
                overall_nonneg_violation = max(nonneg_violation_cur, overall_nonneg_violation)
                overall_l1_violation = max(l1_violation_cur, overall_l1_violation)
                overall_suboptimality = max(suboptimality_cur, overall_suboptimality)

                print(
                    "Nonnegativety violation {:0.12f}, l1 violation {:0.12f}, suboptimality {:0.12f}".format(
                        nonneg_violation, l1_violation, suboptimality
                    )
                )

    print("")
    print("")
    print("")
    print("Overall summary:")
    print("")
    print("Nonnegativety violation {:0.12f} detected.".format(overall_nonneg_violation))
    print("l1 violation violation {:0.12f} detected.".format(overall_l1_violation))
    print("suboptimality {:0.12f} detected.".format(overall_suboptimality))
    print("")
    print("All these values should be close to zero.")
    print("")
