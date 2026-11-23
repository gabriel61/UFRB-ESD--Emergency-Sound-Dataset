using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

public static class AnaliseBootstrapAgrupado
{
    private sealed class Linha
    {
        public int Amostra;
        public string Grupo;
        public int Real;
        public int Prevista;
        public double[] Probabilidades;
    }

    private sealed class Modelo
    {
        public string Nome;
        public Linha[] Linhas;
        public int[][] OrdenacaoAuc;
    }

    private static readonly string[] Metricas = { "accuracy", "f1_weighted", "f1_macro", "auc_macro_ovr" };

    public static int Main(string[] args)
    {
        if (args.Length < 3)
        {
            Console.Error.WriteLine("Uso: analise_bootstrap_agrupado.exe oof_predictions.csv saida_modelos.csv saida_pares.csv [replicacoes]");
            return 1;
        }

        int replicacoes = args.Length >= 4 ? int.Parse(args[3], CultureInfo.InvariantCulture) : 5000;
        var modelos = LerPredicoes(args[0]);
        ValidarAlinhamento(modelos);

        var grupos = modelos[0].Linhas.Select(x => x.Grupo).Distinct().OrderBy(x => x).ToArray();
        var indiceGrupo = grupos.Select((g, i) => new { g, i }).ToDictionary(x => x.g, x => x.i);
        int[] grupoPorLinha = modelos[0].Linhas.Select(x => indiceGrupo[x.Grupo]).ToArray();
        int nClasses = modelos[0].Linhas[0].Probabilidades.Length;

        foreach (var modelo in modelos)
            modelo.OrdenacaoAuc = PrepararOrdenacaoAuc(modelo.Linhas, nClasses);

        var estimativas = new Dictionary<string, double[]>();
        var distribuicoes = new Dictionary<string, double[][]>();
        int[] pesosUnitarios = Enumerable.Repeat(1, grupos.Length).ToArray();
        foreach (var modelo in modelos)
        {
            estimativas[modelo.Nome] = CalcularMetricas(modelo, grupoPorLinha, pesosUnitarios, nClasses);
            distribuicoes[modelo.Nome] = Enumerable.Range(0, Metricas.Length)
                .Select(_ => new double[replicacoes]).ToArray();
        }

        var random = new Random(42);
        int[] pesosGrupo = new int[grupos.Length];
        for (int b = 0; b < replicacoes; b++)
        {
            Array.Clear(pesosGrupo, 0, pesosGrupo.Length);
            for (int i = 0; i < grupos.Length; i++)
                pesosGrupo[random.Next(grupos.Length)]++;

            foreach (var modelo in modelos)
            {
                double[] valores = CalcularMetricas(modelo, grupoPorLinha, pesosGrupo, nClasses);
                for (int m = 0; m < Metricas.Length; m++)
                    distribuicoes[modelo.Nome][m][b] = valores[m];
            }
        }

        EscreverModelos(args[1], modelos, estimativas, distribuicoes, replicacoes, grupos.Length);
        EscreverPares(args[2], modelos, estimativas, distribuicoes, replicacoes, grupos.Length);
        Console.WriteLine("Bootstrap agrupado concluido: {0} modelos, {1} grupos, {2} replicacoes.", modelos.Count, grupos.Length, replicacoes);
        return 0;
    }

    private static List<Modelo> LerPredicoes(string caminho)
    {
        string[] linhas = File.ReadAllLines(caminho);
        string[] cabecalho = linhas[0].Split(',');
        int iAmostra = Array.IndexOf(cabecalho, "sample_index");
        int iModelo = Array.IndexOf(cabecalho, "model");
        int iGrupo = Array.IndexOf(cabecalho, "group_id");
        int iReal = Array.IndexOf(cabecalho, "true_index");
        int iPrevista = Array.IndexOf(cabecalho, "predicted_index");
        int primeiraProb = Array.FindIndex(cabecalho, x => x.StartsWith("prob_", StringComparison.Ordinal));
        int nProb = cabecalho.Length - primeiraProb;
        var porModelo = new Dictionary<string, List<Linha>>();

        for (int i = 1; i < linhas.Length; i++)
        {
            if (string.IsNullOrWhiteSpace(linhas[i])) continue;
            string[] c = linhas[i].Split(',');
            string nome = c[iModelo];
            List<Linha> lista;
            if (!porModelo.TryGetValue(nome, out lista))
            {
                lista = new List<Linha>();
                porModelo[nome] = lista;
            }
            var probs = new double[nProb];
            for (int k = 0; k < nProb; k++)
                probs[k] = double.Parse(c[primeiraProb + k], CultureInfo.InvariantCulture);
            lista.Add(new Linha
            {
                Amostra = int.Parse(c[iAmostra], CultureInfo.InvariantCulture),
                Grupo = c[iGrupo],
                Real = int.Parse(c[iReal], CultureInfo.InvariantCulture),
                Prevista = int.Parse(c[iPrevista], CultureInfo.InvariantCulture),
                Probabilidades = probs
            });
        }

        return porModelo.OrderBy(x => x.Key).Select(x => new Modelo
        {
            Nome = x.Key,
            Linhas = x.Value.OrderBy(y => y.Amostra).ToArray()
        }).ToList();
    }

    private static void ValidarAlinhamento(List<Modelo> modelos)
    {
        if (modelos.Count < 2) throw new InvalidDataException("Sao necessarios pelo menos dois modelos.");
        var referencia = modelos[0].Linhas;
        foreach (var modelo in modelos.Skip(1))
        {
            if (modelo.Linhas.Length != referencia.Length) throw new InvalidDataException("Numero de previsoes diferente entre modelos.");
            for (int i = 0; i < referencia.Length; i++)
                if (modelo.Linhas[i].Amostra != referencia[i].Amostra || modelo.Linhas[i].Grupo != referencia[i].Grupo || modelo.Linhas[i].Real != referencia[i].Real)
                    throw new InvalidDataException("Predicoes OOF nao estao alinhadas entre os modelos.");
        }
    }

    private static int[][] PrepararOrdenacaoAuc(Linha[] linhas, int nClasses)
    {
        var resultado = new int[nClasses][];
        for (int classe = 0; classe < nClasses; classe++)
        {
            int c = classe;
            resultado[classe] = Enumerable.Range(0, linhas.Length)
                .OrderBy(i => linhas[i].Probabilidades[c]).ToArray();
        }
        return resultado;
    }

    private static double[] CalcularMetricas(Modelo modelo, int[] grupoPorLinha, int[] pesosGrupo, int nClasses)
    {
        double[,] cm = new double[nClasses, nClasses];
        double total = 0, corretos = 0;
        for (int i = 0; i < modelo.Linhas.Length; i++)
        {
            int peso = pesosGrupo[grupoPorLinha[i]];
            if (peso == 0) continue;
            Linha l = modelo.Linhas[i];
            cm[l.Real, l.Prevista] += peso;
            total += peso;
            if (l.Real == l.Prevista) corretos += peso;
        }

        double f1Macro = 0, f1Ponderado = 0;
        for (int c = 0; c < nClasses; c++)
        {
            double tp = cm[c, c], suporte = 0, previstos = 0;
            for (int j = 0; j < nClasses; j++)
            {
                suporte += cm[c, j];
                previstos += cm[j, c];
            }
            double precisao = previstos > 0 ? tp / previstos : 0;
            double recall = suporte > 0 ? tp / suporte : 0;
            double f1 = precisao + recall > 0 ? 2 * precisao * recall / (precisao + recall) : 0;
            f1Macro += f1;
            f1Ponderado += suporte * f1;
        }

        double aucMacro = 0;
        for (int c = 0; c < nClasses; c++)
            aucMacro += CalcularAuc(modelo, c, grupoPorLinha, pesosGrupo);

        return new[] { corretos / total, f1Ponderado / total, f1Macro / nClasses, aucMacro / nClasses };
    }

    private static double CalcularAuc(Modelo modelo, int classe, int[] grupoPorLinha, int[] pesosGrupo)
    {
        int[] ordem = modelo.OrdenacaoAuc[classe];
        double positivos = 0, negativos = 0;
        for (int i = 0; i < modelo.Linhas.Length; i++)
        {
            int peso = pesosGrupo[grupoPorLinha[i]];
            if (modelo.Linhas[i].Real == classe) positivos += peso; else negativos += peso;
        }
        if (positivos == 0 || negativos == 0) return double.NaN;

        double negativosAnteriores = 0, favoraveis = 0;
        int inicio = 0;
        while (inicio < ordem.Length)
        {
            double score = modelo.Linhas[ordem[inicio]].Probabilidades[classe];
            int fim = inicio;
            double positivosEmpate = 0, negativosEmpate = 0;
            while (fim < ordem.Length && modelo.Linhas[ordem[fim]].Probabilidades[classe] == score)
            {
                int indice = ordem[fim];
                int peso = pesosGrupo[grupoPorLinha[indice]];
                if (modelo.Linhas[indice].Real == classe) positivosEmpate += peso; else negativosEmpate += peso;
                fim++;
            }
            favoraveis += positivosEmpate * (negativosAnteriores + 0.5 * negativosEmpate);
            negativosAnteriores += negativosEmpate;
            inicio = fim;
        }
        return favoraveis / (positivos * negativos);
    }

    private static void EscreverModelos(string caminho, List<Modelo> modelos, Dictionary<string, double[]> estimativas, Dictionary<string, double[][]> dist, int b, int grupos)
    {
        using (var w = new StreamWriter(caminho, false))
        {
            w.WriteLine("model,metric,estimate,ci95_low,ci95_high,n_groups,n_bootstrap,seed");
            foreach (var modelo in modelos)
                for (int m = 0; m < Metricas.Length; m++)
                    w.WriteLine(string.Join(",", modelo.Nome, Metricas[m], F(estimativas[modelo.Nome][m]), F(Percentil(dist[modelo.Nome][m], 0.025)), F(Percentil(dist[modelo.Nome][m], 0.975)), grupos, b, 42));
        }
    }

    private static void EscreverPares(string caminho, List<Modelo> modelos, Dictionary<string, double[]> estimativas, Dictionary<string, double[][]> dist, int b, int grupos)
    {
        using (var w = new StreamWriter(caminho, false))
        {
            w.WriteLine("metric,model_a,model_b,difference,ci95_low,ci95_high,includes_zero,ci95_bonferroni_low,ci95_bonferroni_high,includes_zero_bonferroni,n_groups,n_bootstrap,seed");
            for (int m = 0; m < Metricas.Length; m++)
                for (int a = 0; a < modelos.Count; a++)
                    for (int c = a + 1; c < modelos.Count; c++)
                    {
                        var diferencas = new double[b];
                        for (int i = 0; i < b; i++) diferencas[i] = dist[modelos[a].Nome][m][i] - dist[modelos[c].Nome][m][i];
                        double baixo = Percentil(diferencas, 0.025), alto = Percentil(diferencas, 0.975);
                        double alphaBonferroni = 0.05 / 6.0;
                        double baixoBonf = Percentil(diferencas, alphaBonferroni / 2.0);
                        double altoBonf = Percentil(diferencas, 1.0 - alphaBonferroni / 2.0);
                        w.WriteLine(string.Join(",", Metricas[m], modelos[a].Nome, modelos[c].Nome,
                            F(estimativas[modelos[a].Nome][m] - estimativas[modelos[c].Nome][m]), F(baixo), F(alto), baixo <= 0 && alto >= 0 ? "1" : "0",
                            F(baixoBonf), F(altoBonf), baixoBonf <= 0 && altoBonf >= 0 ? "1" : "0", grupos, b, 42));
                    }
        }
    }

    private static double Percentil(double[] valores, double p)
    {
        double[] copia = (double[])valores.Clone();
        Array.Sort(copia);
        double pos = p * (copia.Length - 1);
        int baixo = (int)Math.Floor(pos), alto = (int)Math.Ceiling(pos);
        if (baixo == alto) return copia[baixo];
        return copia[baixo] + (pos - baixo) * (copia[alto] - copia[baixo]);
    }

    private static string F(double valor) { return valor.ToString("0.########", CultureInfo.InvariantCulture); }
}
