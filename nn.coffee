class Synapse
  constructor: (@source_neuron, @dest_neuron)->
    @weight = @prev_weight = Math.random() * 2 - 1

class Neuron
  @LEARNING_RATE = 1.0
  @MOMENTUM = 0.05

  constructor: ->
    @prev_threshold = @threshold = Math.random() * 2 - 1
    @synapses_in = []
    @synapses_out = []
    @output = 0.0
    @error = 0.0

  calculate_output: ->
    activation = 0

    for s in @synapses_in
      activation += s.weight * s.source_neuron.output

    activation -= @threshold
    @output = 1.0 / (1.0 + Math.exp(-activation))

  derivative: ->
    @output * (1 - @output)
  
  output_train: (rate, target)->
    @error = (target - @output) * @derivative()
    @update_weights(rate)

  hidden_train: (rate)->
    @error = 0.0
    
    for synapse in @synapses_out
      @error += synapse.prev_weight * synapse.dest_neuron.error

    @error *= @derivative()
    @update_weights(rate)

  update_weights: (rate)->
    for synapse in @synapses_in
      temp_weight = synapse.weight
      synapse.weight += (rate * Neuron.LEARNING_RATE * @error * synapse.source_neuron.output) + (Neuron.MOMENTUM * ( synapse.weight - synapse.prev_weight))
      synapse.prev_weight = temp_weight

    temp_threshold = @threshold
    @threshold += (rate * Neuron.LEARNING_RATE * @error * -1) + (Neuron.MOMENTUM * (@threshold - @prev_threshold))
    @prev_threshold = temp_threshold

class NeuralNetwork
  constructor: (input, hiddens..., output)->
    @input_layer = (new Neuron() for i in [0...input])
    
    @hidden_layers = for hidden in hiddens
      (new Neuron() for i in [0...hidden])

    @output_layer = (new Neuron() for i in [0...output])

    for i in @input_layer
      for h in @hidden_layers[0]
        synapse = new Synapse(i, h)
        i.synapses_out.push synapse
        h.synapses_in.push synapse
    
    for layer, ind in @hidden_layers
      next_layer = if ind==(@hidden_layers.length-1)
        @output_layer
      else
        @hidden_layers[ind+1]

      for h in layer
        for o in next_layer
          synapse = new Synapse(h, o)
          h.synapses_out.push synapse
          o.synapses_in.push synapse

  train: (input, output)->
    @feed_forward(input)

    for neuron, ind in @output_layer
      neuron.output_train 0.5, output[ind]

    for layer in @hidden_layers by -1
      for neuron in layer
        neuron.hidden_train 0.5

  feed_forward: (input)->
    for n, ind in @input_layer
      n.output = input[ind]

    for layer in @hidden_layers
      for n in layer
        n.calculate_output()

    for n in @output_layer
      n.calculate_output()

  current_outputs: ->
    (n.output for n in @output_layer)

nn = new NeuralNetwork(2, 10, 10, 1)

for i in [0...10000]
  nn.train [1, 0], [1]
  nn.train [0, 0], [0]
  nn.train [0, 1], [1]
  nn.train [1, 1], [0]

for i in [[0, 1], [1, 0], [0, 0], [1, 1]]
  nn.feed_forward i
  console.log nn.current_outputs()