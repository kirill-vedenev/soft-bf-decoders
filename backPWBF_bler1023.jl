using Random, Distributions

# Code
const q = 2
const n = 1023
const k = 781
const cosets = [1, 53, 123, 341]
hh = Int[] 
for i in cosets
    t = i
    while true
        append!(hh, t)
        t = (t * q) % n
        t == i && break
    end
end
const h = hh # sparse parity--check polynomial (stored as supp)
const hT = [n - i for i in h] # reciprocal of (h)
const g = [0; h] # generating polynomial (stored as supp)

function weight(x)
    return count(a -> !iszero(a), x)
end
const gamma = weight(h)


# Decoder parameters
const beta = 1.8
const th = 8
const a = 0.3
const b = 1.2
const max_ttl = 5
# Simulation parameters
const snrs = 1.5:0.1:5
const num_it = 25




# The decoder
function sbconv(x, sy, res=zero(x))
    n = length(res)
    for i in 0:n-1
        for j in sy
            if i >= j
                res[i+1] = xor(res[i+1], x[i-j+1])
            else
                res[i+1] = xor(res[i+1], x[n+i-j+1])
            end
        end
    end
    return res
end
function wupc(y, rels, w1, w2, pc=h, res=zero(rels))    
    n = length(y)
    W = 0
    for i in 0:n-1
        si = 0
        for j in pc
            ind = mod(i-j, n) + 1
            si = xor(si, y[ind])
        end
        W += si
        for j in pc
            ind = mod(i-j, n) + 1
            if rels[ind] != w1[i+1]
                res[ind] += (2*si-1)*w1[i+1]
            else
                res[ind] += (2*si-1)*w2[i+1]
            end
        end        
    end
    return W, res
end
function minweights(rels, sy, w1=zero(rels), w2=zero(rels))
    n = length(rels)
    for i in 0:n-1
        t1 = 10
        t2 = 10
        for j in sy
            ind = mod(i-j, n) + 1
            if (rels[ind] <= t1)
                t2 = t1
                t1 = rels[ind]
            elseif (rels[ind] < t2)
                t2 = rels[ind]
            end
        end
        w1[i+1] = t1
        w2[i+1] = t2
    end
    return w1, w2
end
function maxconv(E, sy, s, res=zeros(Int, n))
    n = length(res)    
    for i in 0:n-1
        m = -100; argm = 0
        for j in sy
            pos = mod(i-j, n)
            if E[1 + pos] > m
                m = E[1 + pos]
                argm = pos
            end
        end
        res[1+ argm] += s[1+ i] 
    end
    return res
end
function pwbf(r, beta, th=4, num_it=20)    
    y0 = (Int.(sign.(r)) .+ 1) .>> 1 #hard decision
    y = copy(y0)
    upc = zeros(Int, n); s = sbconv(y, h);
    rels = abs.(r); w1, w2 = minweights(rels, h)
    for ii in 1:num_it
        s = sbconv(y, h);
        iszero(s) && return y
        E = (-beta) * rels
        w, _ = wupc(y, rels, w1, w2, h, E)   
        D = maxconv(E, h, s)
        for i in 1:n
            if (D[i] >= th)
                y[i] = xor(y[i], 1)
            end
        end
    end
    return y0
end
function backpwbf(r, beta, th, a, b, max_ttl, num_it)    
    y0 = (Int.(sign.(r)) .+ 1) .>> 1 #hard decision
    y = copy(y0)
    s = sbconv(y, h);
    rels = abs.(r); w1, w2 = minweights(rels, h)
    ttl = ones(Int, n) * (num_it+1)
    for ii in 1:num_it
        for i in 1:n
            if ttl[i] == ii
                y[i] = y0[i]
            end
        end        
        s = sbconv(y, h);
        iszero(s) && return y
        E = (-beta) * rels
        w, _ = wupc(y, rels, w1, w2, h, E)   
        D = maxconv(E, h, s)
        for i in 1:n
            if (D[i] >= th)
                y[i] = xor(y[i], 1)
                ttl[i] = (ii + 1) + min(max_ttl, trunc(a*(D[i] - th) + b))
            end
        end
        s = sbconv(y, h);
        iszero(s) && return y
    end
    return y0
end





# Simulation
function sigma(snr)
    linsnr = 10^(snr / 10)
    return sqrt(1 / (2 * k / n * linsnr))
end
function testsnr(snr, beta, th, a, b, max_ttl, num_it)
    awgn = Normal(0, sigma(snr))
    bits = (0, 1)
    errors = Threads.Atomic{Int}(0)
    num_trials = 0
    while (errors[] < 250)
        Threads.@threads for j in 1:50000
            c = sbconv(rand(bits, n), g)
            z = rand(awgn, n) + 2 * c .- 1
            d = backpwbf(z, beta, th, a, b, max_ttl, num_it) 
            (c != d) && (Threads.atomic_add!(errors, 1))            
        end        
        num_trials += 50000
        (num_trials >= 10^8) && break
    end
    return errors[] / (num_trials)
end



if (!isempty(ARGS))
    file = open((ARGS[1]), "w")
end
for snr in snrs
    wer = testsnr(snr, beta, th, a, b, max_ttl, num_it)
    println("$snr \t $wer")
    if (!isempty(ARGS))
        println(file, "$snr \t $wer")
        flush(file)
    end
    if wer < 10^-6
        break
    end
end