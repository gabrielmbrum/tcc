/*
 * SSN "full" (conexões intracanal e intercanais) seguindo Ribas et al. (2024).
 *
 * Compilação:
 *   g++ -O3 -shared -fPIC -fopenmp -o SSNModel.so SSNModel.cpp
 *
 * Convenções (Seção 3 do artigo):
 *  - aresta entre vértices a distância euclidiana d, com 0 <= d <= r_max; d = 0 liga os canais
 *    de um MESMO pixel (só entre canais diferentes, sem laços) e pertence ao raio 1;
 *  - direção: do pixel de MENOR intensidade para o de MAIOR (empate: nos dois sentidos),
 *    então o grau de saída de v conta os vizinhos de intensidade >= à de v;
 *  - peso da aresta no raio r:
 *        a = ((|p_i - p_j| + 1) * (d + 1) - 1) / ((L + 1) * (r + 1) - 1)
 *    Como o denominador só depende de r, aqui se acumula apenas o NUMERADOR por faixa de raio
 *    (r-1 < d <= r); a divisão e a soma cumulativa em r ficam no Python.
 *
 * Saída: arrays de tamanho (h * w * z * radius_max), layout (H, W, C, R).
 */
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

typedef struct {
    float *degree_out;  // nº de arestas de saída por faixa de raio
    float *degree_in;
    float *num_out;     // soma dos numeradores (|dp|+1)(d+1)-1 das arestas de saída
    float *num_in;
} NetworkMeasures;

extern "C" NetworkMeasures* model_full(uint8_t* im, int w, int h, int z, int radius_max) {
    const long output_size = (long)w * h * z * radius_max;

    NetworkMeasures *out = (NetworkMeasures*)malloc(sizeof(NetworkMeasures));
    out->degree_out = new float[output_size]();  // () zera
    out->degree_in  = new float[output_size]();
    out->num_out    = new float[output_size]();
    out->num_in     = new float[output_size]();

    const int r_max_sq = radius_max * radius_max;
    std::vector<int> vraio(r_max_sq + 1, 0);  // d^2 -> faixa de raio (r-1)^2 < d^2 <= r^2
    for (int i = 1; i <= r_max_sq; i++) {
        for (int r = 1; r <= radius_max; r++) {
            if ((r - 1) * (r - 1) < i && i <= r * r) { vraio[i] = r; break; }
        }
    }

    #pragma omp parallel for collapse(3)
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < z; c++) {
                const long central_idx = ((long)y * w + x) * z + c;
                for (int i = x - radius_max; i <= x + radius_max; i++) {
                    for (int j = y - radius_max; j <= y + radius_max; j++) {
                        if (i < 0 || i >= w || j < 0 || j >= h) continue;
                        const int d_sq = (x - i) * (x - i) + (y - j) * (y - j);
                        if (d_sq > r_max_sq) continue;
                        const int rr_idx = (d_sq == 0) ? 0 : vraio[d_sq] - 1;  // d = 0 -> raio 1
                        const float d = sqrtf((float)d_sq);
                        const long out_idx = central_idx * radius_max + rr_idx;

                        for (int cw = 0; cw < z; cw++) {
                            if (d_sq == 0 && cw == c) continue;  // sem laços
                            const long neighbor_idx = ((long)j * w + i) * z + cw;
                            // diff < 0: vizinho mais claro -> aresta SAI do vértice central
                            const float diff = (float)im[central_idx] - (float)im[neighbor_idx];
                            const float abs_diff = (diff < 0) ? -diff : diff;
                            const float num = (abs_diff + 1.0f) * (d + 1.0f) - 1.0f;

                            if (diff <= 0) { out->degree_out[out_idx] += 1.0f; out->num_out[out_idx] += num; }
                            if (diff >= 0) { out->degree_in[out_idx]  += 1.0f; out->num_in[out_idx]  += num; }
                        }
                    }
                }
            }
        }
    }
    return out;
}

extern "C" void free_network_measures(NetworkMeasures *nm) {
    if (nm) {
        delete[] nm->degree_out;
        delete[] nm->degree_in;
        delete[] nm->num_out;
        delete[] nm->num_in;
        free(nm);
    }
}
