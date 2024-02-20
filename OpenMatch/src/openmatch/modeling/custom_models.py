import torch
from transformers import T5Model, T5Config

from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqModelOutput,
)

import warnings
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

class T5ModelWithFusion(T5Model):

    def __init__(self, config: T5Config):
        super().__init__(config)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        fusion=None
        ):

        use_cache = False

        aggregator_window = False

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        def revert_concatenated_tensor(concatenated_tensor):
            N_by_4, full_dim = concatenated_tensor.shape
            reverted_tensor = concatenated_tensor.reshape(N_by_4, -1, full_dim//4)
            N = N_by_4 * 4
            reverted_tensor = reverted_tensor.view(N, full_dim//4)
            return reverted_tensor

        if fusion:
            input_ids = revert_concatenated_tensor(input_ids)
            attention_mask = revert_concatenated_tensor(attention_mask)

            


        def custom_interleave(tensor1, fusion):
            n_docs, dim = tensor1.shape

            n_docs_fusion = n_docs // fusion

            tensor2 = tensor1.view(int(n_docs/fusion), fusion, dim)[:, :, :int(dim/fusion)].reshape(int(n_docs/fusion), dim)

            interleaved = torch.empty((n_docs + n_docs_fusion, dim), dtype=tensor1.dtype)

            interleaved[::fusion + 1] = tensor2[:n_docs_fusion]

            idx = 0
            for i in range(n_docs_fusion):
                interleaved[idx + 1:idx + fusion + 1] = tensor1[i * fusion: (i + 1) * fusion]
                idx += fusion + 1

            return interleaved

        if aggregator_window and fusion:
            device = input_ids.device
            new_input_ids = custom_interleave(input_ids, fusion)
            new_attention_mask = custom_interleave(attention_mask, fusion)
            
            del input_ids
            del attention_mask

            input_ids = new_input_ids.to(device)
            attention_mask = new_attention_mask.to(device)
            

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        
        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        
        if fusion:
            fusion_parameter = fusion if not aggregator_window else fusion+1
            dim = hidden_states.size(1)
            len_decoding = len(decoder_input_ids)
            #fused_samples = int(len_decoding/fusion)
            fused_samples = int(len_decoding)
            fused_dimensionality = int(dim*fusion_parameter)
            

            hidden_states=hidden_states.view(fused_samples, fused_dimensionality, hidden_states.size(-1))
            attention_mask=attention_mask.view(fused_samples, fused_dimensionality)
            #decoder_input_ids = decoder_input_ids[list(range(0,len_decoding, fusion))]
            
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )