
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_hcth_a_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_hcth_a", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.112226577874593e-01, -1.112220390654026e-01, -1.112202806267085e-01, -1.112303916394596e-01, -1.112223231210877e-01, -1.112223231210877e-01, -8.099049847291537e-02, -8.098779303474536e-02, -8.091412950763331e-02, -8.104521445273778e-02, -8.098919432913911e-02, -8.098919432913911e-02, -5.405485893411146e-02, -5.409847220422091e-02, -5.447166426444456e-02, -5.462340873596803e-02, -5.407070440005551e-02, -5.407070440005551e-02, -2.741832429380239e-02, -2.843720385911057e-02, -5.497242046235481e-02, -3.249622784499888e-03, -2.774465418151071e-02, -2.774465418151071e-02, 1.088567173510090e-02, 1.131348949947147e-02, 2.759008305483904e-02, 5.684132574165428e-03, 1.121125101708241e-02, 1.121125101708241e-02, -7.718432932213766e-02, -7.704596985399256e-02, -7.717054072644489e-02, -7.706295720694578e-02, -7.711295937266542e-02, -7.711295937266542e-02, -7.783833924092423e-02, -7.797253439035866e-02, -7.777990037798969e-02, -7.790324449877027e-02, -7.794193548768741e-02, -7.794193548768741e-02, -4.504570981886884e-02, -3.472256156054628e-02, -4.481482803942505e-02, -3.639649236013692e-02, -4.428357432485814e-02, -4.428357432485814e-02, 1.802200290516571e-02, -1.557881408857328e-02, 1.645521205865256e-02, -4.977194384630800e-02, 6.631122326268142e-03, 6.631122326268142e-03, 5.497243994071531e-03, 6.218036664473754e-03, 4.748497065387299e-03, 2.672991235262567e-02, 5.702550551521465e-03, 5.702550551521465e-03, -3.035764684024483e-02, -3.335025631566372e-02, -3.223744776966412e-02, -3.140703950458465e-02, -3.181792130118324e-02, -3.181792130118324e-02, -2.949300998825602e-02, -4.733493789673845e-02, -4.456868627715432e-02, -4.002548044994809e-02, -4.254344509113245e-02, -4.254344509113243e-02, -3.584277284648235e-02, -2.749890031722219e-02, -3.638381978070350e-02, -4.196457612353428e-02, -4.056139386410423e-02, -4.056139386410424e-02, -4.593862572515295e-02, 2.755920173837557e-02, 2.862665723356533e-02, -3.899187136304288e-02, 2.011056190442639e-02, 2.011056190442637e-02, 1.244522162966357e-02, 1.799205799227883e-03, 3.360230596644454e-03, 1.930402989560980e-02, 4.898572165641455e-03, 4.898572165641451e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_hcth_a_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_hcth_a", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-4.903833610197895e-02, -4.903771145242042e-02, -4.903586840769350e-02, -4.904608075457659e-02, -4.903799879282186e-02, -4.903799879282186e-02, -4.323815529679465e-02, -4.323299952026071e-02, -4.309200652462195e-02, -4.333730838016736e-02, -4.323554324247095e-02, -4.323554324247095e-02, -4.576713129019057e-02, -4.623316842032694e-02, -6.082331036611413e-02, -6.008307370811036e-02, -4.593435538029405e-02, -4.593435538029405e-02, -6.783909083973595e-02, -6.506208630325278e-02, -3.620621805143318e-02, -8.219987159035438e-02, -6.696990413289625e-02, -6.696990413289625e-02, 1.350697185777513e-02, 1.396448828658430e-02, 1.644015294852898e-02, 7.395788625771082e-03, 1.384319734272309e-02, 1.384319734272309e-02, -3.465068078302482e-02, -3.461278052244079e-02, -3.464680140380801e-02, -3.461731082615725e-02, -3.463083903894529e-02, -3.463083903894529e-02, -8.355805432020846e-02, -8.191826085308862e-02, -8.606734432571968e-02, -8.474642007366742e-02, -7.929506929928902e-02, -7.929506929928902e-02, -2.433916777922582e-02, -2.430839362268205e-02, -2.492252182462339e-02, -2.073079918342512e-02, -2.289236016298886e-02, -2.289236016298886e-02, -4.785046344328830e-02, -9.221070592541540e-02, -5.000470288733026e-02, -3.320265526099940e-02, -7.004208007143020e-02, -7.004208007143020e-02, 7.159063288204689e-03, 8.061543279585201e-03, 6.187683861294714e-03, -4.122962331696165e-03, 7.404305035162607e-03, 7.404305035162616e-03, -3.022308722701466e-02, -2.485804028434237e-02, -2.686164382190418e-02, -2.847337250074762e-02, -2.767555992287887e-02, -2.767555992287887e-02, -3.038520128305743e-02, -3.459526686118761e-02, -2.549033127348076e-02, -2.017391980082330e-02, -2.225724790122782e-02, -2.225724790122782e-02, -2.398142086321178e-02, -8.546058438627864e-02, -6.734295152406923e-02, -3.587604848364635e-02, -4.984074054649428e-02, -4.984074054649430e-02, -3.552641929359193e-02, 2.052463357476598e-02, 7.314992720266589e-03, -2.638868201815166e-02, -3.384146807690247e-02, -3.384146807690248e-02, 1.525776926114817e-02, 2.387887386680591e-03, 4.425200295147338e-03, -3.332773377341201e-02, 6.384431514895813e-03, 6.384431514895808e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_hcth_a_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_hcth_a", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.745235571544795e-10, -1.745258595491001e-10, -1.745283498657437e-10, -1.744908035337332e-10, -1.745248364006152e-10, -1.745248364006152e-10, -7.135279452642406e-07, -7.136732148381011e-07, -7.175329118374930e-07, -7.101763306713841e-07, -7.135882441910625e-07, -7.135882441910625e-07, -5.042221555411877e-04, -4.869165551940180e-04, -8.011524873547594e-06, -3.278164881587104e-05, -4.980023959544725e-04, -4.980023959544725e-04, 2.798319924359193e-01, 2.489954283949780e-01, -4.855945522257147e-04, 1.179540969412695e+00, 2.703101853662379e-01, 2.703101853662379e-01, 2.461314504499788e+01, 2.449993094526453e+01, 7.007778976939699e+00, 1.903059351650271e+01, 2.549107093702516e+01, 2.549107093702516e+01, -1.965845837095234e-07, -1.974898157564682e-07, -1.966740902200284e-07, -1.973779975294983e-07, -1.970529592634115e-07, -1.970529592634115e-07, -1.959160815528181e-07, -3.959697806558115e-07, 9.889291872466820e-08, -6.466173627657616e-08, -7.102486454692733e-07, -7.102486454692733e-07, -3.836848950748683e-03, -5.533418478254975e-03, -4.524988479604088e-03, -7.813863174189993e-03, -3.726796654634031e-03, -3.726796654634031e-03, 2.220541922754523e+00, 3.129905075978652e-01, 2.512908046204847e+00, -3.064387155836980e-05, 1.884634020356948e+00, 1.884634020356948e+00, 2.025150098381554e+01, 2.004874688503097e+01, 5.878859070709517e+01, 5.693713862574398e+00, 2.986122072350708e+01, 2.986122072350556e+01, -6.568601663186240e-03, -7.201805076357380e-03, -6.981271688728958e-03, -6.686687302729167e-03, -6.840848772222005e-03, -6.840848772222005e-03, -8.320640144521435e-03, -3.247525548064360e-03, -5.238082424464506e-03, -7.465521472964817e-03, -6.315824598300926e-03, -6.315824598300923e-03, -4.415284348031449e-03, 1.069411803274263e-01, 3.306428556866864e-02, -1.028931901294527e-02, 3.563280599021546e-03, 3.563280599021573e-03, -4.419498639500996e-03, 6.164308111421653e+00, 5.302378968233697e+00, -2.919913566833966e-02, 4.366389452798370e+00, 4.366389452798372e+00, 1.803365267561968e+01, 3.724557302498307e+01, 3.117713754084311e+01, 5.444811035429233e+00, 4.420081071307909e+01, 4.420081071308469e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05