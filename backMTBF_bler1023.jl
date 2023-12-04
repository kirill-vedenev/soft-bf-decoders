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

#decoder parameters
const minT = (gamma+1) >> 1
const maxT = gamma - 1
const alpha = 0.07
const tstep = 9
const max_ttl = 12
const a = 3.0
const b = 2.6
const snrs = 1.5:0.1:5


# Decoder
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

function UPC(x, sy=h, res=zero(x))
    n = length(x)
    si = 0
    for i in 0:n-1
        si = 0
        for j in sy
            if i >= j
                si = xor(si, x[i-j+1])
            else
                si = xor(si, x[n+i-j+1])
            end
        end
        if (si == 1)
            for j in sy
                if i >= j
                    res[i-j+1] += 1
                else
                    res[n+i-j+1] += 1
                end
            end
        end
    end
    return res
end

function backMTBF(r, alpha, tstep, a, b, num_it, minT, maxT, maxttl)
    y0 = (Int.(sign.(r)) .+ 1) .>> 1 #hard decision    
    th = min.(trunc.(Int, abs.(r / alpha)) .+ minT, maxT) #thresholds
    ttl = ones(Int, n) * (num_it + 1)
    y = copy(y0)
    upc = zeros(Int, n)
    s = zeros(Int, n)
    for i in 1:num_it
        for j in 1:n
            if ttl[j] == i
                y[j] = y0[j]
            end
        end
        fill!(upc, 0)
        UPC(y, h, upc)
        iszero(upc) && return y
        for j in 1:n
            if (y[j] == y0[j]) && (upc[j] >= th[j])
                y[j] = xor(y[j], 1)
                ttl[j] = (i + 1) + min(maxttl, trunc(Int, a * (upc[j] - th[j]) + b))
            elseif (y[j] != y0[j]) && (upc[j] >= minT) && (upc[j] >= th[j] - tstep)
                y[j] = y0[j]
                ttl[j] = 0
            end
        end
        fill!(s, 0)
        sbconv(y, h, s)
        iszero(s) && return y
    end
    return y0
end

# Testing

function sigma(snr)
    linsnr = 10^(snr / 10)
    return sqrt(1 / (2 * k / n * linsnr))
end

function testsnr(snr, alpha, tstep, a, b, num_it, minT, maxT, maxttl)
    linsnr = 10^(snr / 10)
    sigma = sqrt(1 / (2 * k / n * linsnr))
    awgn = Normal(0, sigma)
    bits = (0, 1)
    errors = Threads.Atomic{Int}(0)
    num_trials = 0
    while (errors[] < 250)
        Threads.@threads for j in 1:20000
            c = sbconv(rand(bits, n), g)
            z = rand(awgn, n) + 2 * c .- 1
            d = backMTBF(z, alpha, tstep, a, b, num_it, minT, maxT, maxttl)
            (c != d) && (Threads.atomic_add!(errors, 1))
        end
        num_trials += 20000
    end
    return errors[] / (num_trials)
end


if (!isempty(ARGS))
    file = open((ARGS[1]), "w")
end

const num_it = 50

for snr in snrs
    wer = testsnr(snr, alpha, tstep, a, b, num_it, minT, maxT, max_ttl)
    println("$snr \t $wer")
    if (!isempty(ARGS))
        println(file, "$snr \t $wer")
        flush(file)
    end
    if wer < 10^-6
        break
    end
end
