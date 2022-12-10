from .ar.flow import ArgmaxARFlow
from .coupling.flow import ArgmaxCouplingFlow
from .ar.vorflow import VoronoiARFlow
from .coupling.vorflow import VoronoiCouplingFlow


def get_model(cfg, args, data_shape, num_classes):

    if cfg.dequantization == "argmax":

        if args.model == "ar":
            return ArgmaxARFlow(
                data_shape=data_shape,
                num_classes=num_classes,
                num_steps=args.num_steps,
                actnorm=args.actnorm,
                perm_channel=args.perm_channel,
                perm_length=args.perm_length,
                base_dist=args.base_dist,
                encoder_steps=args.encoder_steps,
                encoder_bins=args.encoder_bins,
                context_size=args.context_size,
                lstm_layers=args.lstm_layers,
                lstm_size=args.lstm_size,
                lstm_dropout=args.lstm_dropout,
                context_lstm_layers=args.context_lstm_layers,
                context_lstm_size=args.context_lstm_size,
                input_dp_rate=args.input_dp_rate,
            )

        elif args.model == "coupling":
            return ArgmaxCouplingFlow(
                data_shape=data_shape,
                num_classes=num_classes,
                num_steps=args.num_steps,
                actnorm=args.actnorm,
                num_mixtures=args.num_mixtures,
                perm_channel=args.perm_channel,
                perm_length=args.perm_length,
                base_dist=args.base_dist,
                encoder_steps=args.encoder_steps,
                encoder_bins=args.encoder_bins,
                encoder_ff_size=args.encoder_ff_size,
                context_size=args.context_size,
                context_ff_layers=args.context_ff_layers,
                context_ff_size=args.context_ff_size,
                context_dropout=args.context_dropout,
                lstm_layers=args.lstm_layers,
                lstm_size=args.lstm_size,
                lstm_dropout=args.lstm_dropout,
                input_dp_rate=args.input_dp_rate,
            )

    elif cfg.dequantization == "voronoi":

        if args.model == "ar":
            return VoronoiARFlow(
                data_shape=data_shape,
                num_classes=num_classes,
                embedding_dim=cfg.embedding_dim,
                num_steps=args.num_steps,
                actnorm=args.actnorm,
                perm_channel=args.perm_channel,
                perm_length=args.perm_length,
                base_dist=args.base_dist,
                encoder_steps=args.encoder_steps,
                encoder_bins=args.encoder_bins,
                context_size=args.context_size,
                lstm_layers=args.lstm_layers,
                lstm_size=args.lstm_size,
                lstm_dropout=args.lstm_dropout,
                context_lstm_layers=args.context_lstm_layers,
                context_lstm_size=args.context_lstm_size,
                input_dp_rate=args.input_dp_rate,
            )

        elif args.model == "coupling":
            return VoronoiCouplingFlow(
                data_shape=data_shape,
                num_classes=num_classes,
                embedding_dim=cfg.embedding_dim,
                num_steps=args.num_steps,
                actnorm=args.actnorm,
                num_mixtures=args.num_mixtures,
                perm_channel=args.perm_channel,
                perm_length=args.perm_length,
                base_dist=args.base_dist,
                encoder_steps=args.encoder_steps,
                encoder_bins=args.encoder_bins,
                encoder_ff_size=args.encoder_ff_size,
                context_size=args.context_size,
                context_ff_layers=args.context_ff_layers,
                context_ff_size=args.context_ff_size,
                context_dropout=args.context_dropout,
                lstm_layers=args.lstm_layers,
                lstm_size=args.lstm_size,
                lstm_dropout=args.lstm_dropout,
                input_dp_rate=args.input_dp_rate,
            )
