REQUISITOS VS0.2

Separar abas de data manegemente e Model Training e Fine-tunning para a versão PRO

Criar aba de cadastro
	-Dados do paciente (rapha e Daniel)
	-Rade treinada do sujeito
	-Caso não tenha cadastro, usar a rede do physionet e cadastrar um novo
	-Caso ja tenha abre a rede do usuário com seus dados

Cada sujeito teremos
	- Um histórico de todas as redes treinadas com suas respectivas datas e acuracia
	- Dados funcionais do sujeito

Criar uma aba de intervensão 
	- um botão para iniciar
	- Ler do VR a mão que deveria ser a correta
	- Ler do EEG o sinal e dentificar qual mão teve imagética, mandar para o VR a mão identificada.
	- Dar feedback de qual mão a rede identificou e qual a real do jogo.