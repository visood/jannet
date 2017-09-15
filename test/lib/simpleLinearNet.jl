importall ANN
logreg = Network(1,
                 Dict(Arrow(UInt8(1), UInt8(0))=>0.01*rand(Float64, 1, 2)),
                 Dict(0x01=>zeros(1)),
                 Dict(0x01=>sigmoid),
                 Dict(0x00=>[],
                      0x01=>[0x00]),
                 Dict(0x00=>[0x01],
                      0x01=>[]))

forwprop(logreg, [0.0, 0.0])

hidden1 = Network(2,
                  Dict(Arrow(0x01, 0x00)=>0.01*rand(Float64, 2, 2),
                       Arrow(0x02, 0x01)=>0.01*rand(Float64, 1, 2)),
                  Dict(0x01=>zeros(2),
                       0x02=>zeros(1)),
                  Dict(0x01=>tanh,
                       0x02=>sigmoid),
                  Dict(0x00=>[],
                       0x01=>[0x00],
                       0x02=>[0x01]),
                  Dict(0x00=>[0x01],
                       0x01=>[0x02],
                       0x02=>[]))

txor = Network(2,
              Dict(Arrow(0x01, 0x00)=>0.01*rand(Float64, 2, 2),
                   Arrow(0x02, 0x01)=>0.01*rand(Float64, 1, 2)),
              Dict(0x01=>zeros(2),
                   0x02=>zeros(1)),
              Dict(0x01=>tanh,
                   0x02=>sigmoid),
              Dict(0x00=>[],
                   0x01=>[0x00],
                   0x02=>[0x01]),
              Dict(0x00=>[0x01],
                   0x01=>[0x02],
                   0x02=>[]))

train!(txor,
       [0.0 0.0 1.0 1.0; 0.0 1.0 0.0 1.0],
       reshape([0.0, 1.0, 1.0, 0.0], (1, 4)),
       100,
       10.0)

xor = Network(2,
              Dict(Arrow(0x01, 0x00)=>[1.0 1.0; 1.0 1.0],
                   Arrow(0x02, 0x01)=>reshape([-1.0, 1.0], (1, 2))),
              Dict(0x01=>[-1.9, 0.0],
                   0x02=>[0.0]),
              Dict(0x01=>sigmoid,
                   0x02=>sigmoid),
              Dict(0x00=>[],
                   0x01=>[0x00],
                   0x02=>[0x01]),
              Dict(0x00=>[0x01],
                   0x01=>[0x02],
                   0x02=>[]))
