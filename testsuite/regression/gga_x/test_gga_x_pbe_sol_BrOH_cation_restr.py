
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pbe_sol_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_sol", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.047429329575011e+01, -2.047433183450133e+01, -2.047455056840682e+01, -2.047397621480671e+01, -2.047427042221396e+01, -2.047427042221396e+01, -3.371783650544747e+00, -3.371763058114313e+00, -3.371369849555759e+00, -3.372694078743562e+00, -3.371840240671059e+00, -3.371840240671059e+00, -6.676200984629154e-01, -6.673917183892714e-01, -6.633967646232906e-01, -6.681965251430013e-01, -6.666125892650629e-01, -6.666125892650629e-01, -1.986619548107972e-01, -1.998226974016153e-01, -7.732135060896133e-01, -1.685100043513522e-01, -1.796065272009512e-01, -1.796065272009513e-01, -1.008264401523888e-02, -1.061079646503729e-02, -5.700345601134093e-02, -5.827373855033942e-03, -7.320561689271223e-03, -7.320561689271223e-03, -4.948755106725143e+00, -4.948755488864735e+00, -4.948762831577070e+00, -4.948763049578953e+00, -4.948751001489582e+00, -4.948751001489582e+00, -2.000397338429413e+00, -2.011167580527141e+00, -1.998396302004024e+00, -2.007881305018995e+00, -2.007342736688995e+00, -2.007342736688995e+00, -5.656458497491723e-01, -5.979411547036659e-01, -5.260111373631152e-01, -5.314919082227065e-01, -5.725776186317852e-01, -5.725776186317852e-01, -1.318627451354034e-01, -2.130656627481705e-01, -1.237054990492846e-01, -1.799727807446253e+00, -1.447665701351090e-01, -1.447665701351090e-01, -4.496700528111971e-03, -5.696700839242691e-03, -4.355242776719369e-03, -8.815495245816321e-02, -5.246744818531863e-03, -5.246744818531863e-03, -5.491858493637096e-01, -5.496306436693272e-01, -5.494785744229624e-01, -5.493449936332855e-01, -5.494115716766844e-01, -5.494115716766844e-01, -5.332533571242677e-01, -4.891958430595283e-01, -5.012063939575464e-01, -5.133400631347051e-01, -5.069728250485165e-01, -5.069728250485165e-01, -6.275132303555117e-01, -2.541621617903391e-01, -2.882271384498996e-01, -3.488786120858578e-01, -3.155330764823026e-01, -3.155330764823026e-01, -4.511149299843224e-01, -5.474744307710133e-02, -7.313371538156900e-02, -3.295697059040050e-01, -1.067920223260033e-01, -1.067920223260033e-01, -1.421749552816645e-02, -1.523225610102202e-03, -3.196942095483637e-03, -1.016300426753932e-01, -4.854975686300430e-03, -4.854975686300425e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pbe_sol_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_sol", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.563142649167216e+01, -2.563151867283801e+01, -2.563191734574768e+01, -2.563054610203376e+01, -2.563126619048627e+01, -2.563126619048627e+01, -4.131720315368085e+00, -4.131758113497035e+00, -4.132733310469445e+00, -4.131822622494737e+00, -4.131865233852604e+00, -4.131865233852604e+00, -7.806233991770831e-01, -7.793314665591237e-01, -7.480390204580710e-01, -7.547430400788491e-01, -7.537141944015132e-01, -7.537141944015132e-01, -1.934208050472618e-01, -1.959904443941703e-01, -9.148598546095300e-01, -1.594549404366178e-01, -1.698125191772723e-01, -1.698125191772724e-01, -1.341426129375701e-02, -1.411267219514744e-02, -7.197556214159039e-02, -7.764445039370016e-03, -9.748712535716573e-03, -9.748712535716573e-03, -6.295991672132240e+00, -6.298213216219903e+00, -6.296093251912763e+00, -6.298054460980084e+00, -6.297118160793236e+00, -6.297118160793236e+00, -2.237963396230107e+00, -2.256692099404432e+00, -2.220572665358412e+00, -2.237037952635241e+00, -2.256829463612576e+00, -2.256829463612576e+00, -7.005251174315460e-01, -7.803549144009144e-01, -6.454265210722798e-01, -6.883441137815592e-01, -7.142011370479014e-01, -7.142011370479014e-01, -1.371142645761688e-01, -2.010867750888044e-01, -1.310187900643239e-01, -2.351569087258406e+00, -1.406989365727421e-01, -1.406989365727421e-01, -5.992759513976790e-03, -7.590141628733498e-03, -5.800876947115258e-03, -1.020631754772105e-01, -6.988368355684742e-03, -6.988368355684742e-03, -7.268230479208786e-01, -7.180282803996586e-01, -7.210992932707372e-01, -7.236541809143755e-01, -7.223740381202726e-01, -7.223740381202726e-01, -7.084968340971093e-01, -5.738207676735441e-01, -6.112991807317864e-01, -6.497477759835306e-01, -6.301258148656074e-01, -6.301258148656074e-01, -8.172812766401424e-01, -2.455191952400264e-01, -2.932901258618339e-01, -4.022325178306311e-01, -3.419113880369968e-01, -3.419113880369968e-01, -5.276455251468865e-01, -6.958334267577196e-02, -8.945516934020119e-02, -3.936035923257760e-01, -1.143676970769958e-01, -1.143676970769958e-01, -1.888301875197720e-02, -2.030736254960825e-03, -4.261102352014838e-03, -1.109788145908488e-01, -6.466808268485481e-03, -6.466808268485474e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pbe_sol_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_sol", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.222177652262097e-09, -4.222140806425488e-09, -4.221946018192244e-09, -4.222494857096498e-09, -4.222211673438633e-09, -4.222211673438633e-09, -5.825352136810099e-06, -5.825434367822513e-06, -5.826762974617980e-06, -5.820029204127445e-06, -5.824889049437110e-06, -5.824889049437110e-06, -3.906926047284342e-03, -3.915447689755426e-03, -4.094006250159189e-03, -3.973820035090659e-03, -4.009445819761480e-03, -4.009445819761480e-03, -5.299461952523643e-01, -5.191215793525004e-01, -2.155446696832606e-03, -8.848940275183702e-01, -7.565316099947834e-01, -7.565316099947831e-01, -4.324245069857511e+00, -4.567054976602716e+00, -2.415881593214507e+00, -3.935446804944788e+00, -4.997159452064834e+00, -4.997159452064847e+00, -1.223034212312456e-06, -1.222725574529322e-06, -1.223013840922483e-06, -1.222741469695691e-06, -1.222881194796181e-06, -1.222881194796181e-06, -4.973463565308086e-05, -4.859900367598377e-05, -5.011674137484669e-05, -4.910703403988811e-05, -4.891746073074166e-05, -4.891746073074166e-05, -7.301983907905049e-03, -5.633148868539480e-03, -9.826260255851583e-03, -9.074868447943536e-03, -6.920445741986845e-03, -6.920445741986845e-03, -1.303794025712054e+00, -3.767856839452794e-01, -1.478986296848460e+00, -6.857657837285006e-05, -1.357996715366817e+00, -1.357996715366817e+00, -5.186612087062384e+00, -4.631591422575532e+00, -2.907711813864359e+01, -2.461187418688752e+00, -1.353140111190637e+01, -1.353140111190634e+01, -7.834073626117976e-03, -7.884582456300889e-03, -7.866787062818213e-03, -7.852316968871686e-03, -7.859583217856257e-03, -7.859583217856257e-03, -8.787324089944223e-03, -1.352588731316826e-02, -1.196917726085779e-02, -1.060202085380486e-02, -1.128762850779732e-02, -1.128762850779732e-02, -4.650993457761299e-03, -1.968752127656062e-01, -1.199177018029357e-01, -5.284020786514726e-02, -8.165779279906932e-02, -8.165779279906933e-02, -1.873767640998302e-02, -2.219530034768598e+00, -2.118982908736400e+00, -6.491581972785890e-02, -2.449640985109934e+00, -2.449640985109936e+00, -3.478876342308573e+00, -2.274311921202609e+01, -1.100435953810926e+01, -2.549492490456899e+00, -1.709205828545668e+01, -1.709205828545673e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05