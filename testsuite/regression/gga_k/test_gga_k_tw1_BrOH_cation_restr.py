
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_tw1_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.192596773248822e+03, 2.192605582735868e+03, 2.192654009282334e+03, 2.192522751132455e+03, 2.192590210192755e+03, 2.192590210192755e+03, 5.925629766293482e+01, 5.925573735555356e+01, 5.924568949900580e+01, 5.928566987808324e+01, 5.925848274029654e+01, 5.925848274029654e+01, 2.299375079466523e+00, 2.297015374400413e+00, 2.246295286553318e+00, 2.280199702693690e+00, 2.270147840615357e+00, 2.270147840615357e+00, 1.845343805312591e-01, 1.882270896510926e-01, 3.093356318240303e+00, 1.186007534247860e-01, 1.425211975135999e-01, 1.425211975135999e-01, 2.998808343457114e-04, 3.321935410122806e-04, 9.962102433704125e-03, 1.000636312288776e-04, 1.579755888746062e-04, 1.579755888746062e-04, 1.283755278403666e+02, 1.283811246102642e+02, 1.283761595116115e+02, 1.283810950012312e+02, 1.283781597063673e+02, 1.283781597063673e+02, 2.037024763182638e+01, 2.061119590762850e+01, 2.028049296230145e+01, 2.049301047643083e+01, 2.054653733686041e+01, 2.054653733686041e+01, 1.670703508370908e+00, 1.879160361876672e+00, 1.442478653731891e+00, 1.483667286748319e+00, 1.713847673130470e+00, 1.713847673130470e+00, 6.278729285961133e-02, 1.985542418476464e-01, 5.425895836118014e-02, 1.702568095953966e+01, 8.228236267331723e-02, 8.228236267331723e-02, 5.957294861756759e-05, 9.562806652250236e-05, 5.590734535345174e-05, 2.548561812079295e-02, 8.113732758076524e-05, 8.113732758076524e-05, 1.586880163375663e+00, 1.587910273023423e+00, 1.587615424960279e+00, 1.587292389873281e+00, 1.587456638857811e+00, 1.587456638857811e+00, 1.496505588190703e+00, 1.235580159300345e+00, 1.308196290689362e+00, 1.380438350374300e+00, 1.342864042745507e+00, 1.342864042745507e+00, 2.069262203192474e+00, 2.990280547936027e-01, 4.038215759738620e-01, 6.255125592898183e-01, 5.006174138359116e-01, 5.006174138359116e-01, 1.049935071907741e+00, 9.144096110188255e-03, 1.679714336455064e-02, 5.631976575166575e-01, 4.001192069827159e-02, 4.001192069827160e-02, 5.970182983473696e-04, 6.834014815314725e-06, 3.010864538239390e-05, 3.559780636025581e-02, 6.947083381457764e-05, 6.947083381457752e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_tw1_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.347038496457656e+03, 3.347059249789893e+03, 3.347153933194348e+03, 3.346845118592405e+03, 3.347006584848331e+03, 3.347006584848331e+03, 8.810204713704813e+01, 8.810281479247551e+01, 8.812489890754286e+01, 8.811998970177932e+01, 8.810721519285040e+01, 8.810721519285040e+01, 3.246965258051951e+00, 3.239354437926632e+00, 3.066337827697238e+00, 3.117223115040757e+00, 3.106202255291734e+00, 3.106202255291734e+00, 2.343984066627444e-01, 2.390802880503054e-01, 4.421787422520343e+00, 1.574665968927072e-01, 1.837730493545479e-01, 1.837730493545479e-01, 4.993252677129520e-04, 5.530553825715649e-04, 1.620120297049056e-02, 1.667221803416235e-04, 2.631506703111152e-04, 2.631506703111152e-04, 2.001333000338761e+02, 2.002363383567461e+02, 2.001381786070441e+02, 2.002291386204675e+02, 2.001854447403858e+02, 2.001854447403858e+02, 2.762310336479675e+01, 2.801888777973838e+01, 2.735001060363363e+01, 2.769552467698612e+01, 2.797727021056125e+01, 2.797727021056125e+01, 2.516129752709066e+00, 3.034663956408312e+00, 2.148097089000762e+00, 2.370121772693980e+00, 2.604211216014397e+00, 2.604211216014397e+00, 9.173340760705877e-02, 2.571858876884635e-01, 8.026099008736177e-02, 2.754184411786147e+01, 1.136077573625709e-01, 1.136077573625709e-01, 9.926768555910617e-05, 1.593300368293919e-04, 9.313601887641004e-05, 3.970751145368838e-02, 1.351672547474655e-04, 1.351672547474655e-04, 2.615483426467717e+00, 2.568024469547618e+00, 2.584456818919101e+00, 2.598231795830842e+00, 2.591315308122603e+00, 2.591315308122603e+00, 2.480936455000700e+00, 1.750404838095111e+00, 1.934284985970292e+00, 2.137473661956267e+00, 2.031433341968993e+00, 2.031433341968993e+00, 3.331948995198866e+00, 3.801957996464842e-01, 5.169900952183822e-01, 8.712127070954668e-01, 6.626466883846159e-01, 6.626466883846158e-01, 1.483113039168177e+00, 1.491626800727446e-02, 2.690812064822916e-02, 8.131661534266114e-01, 5.960952924807977e-02, 5.960952924807977e-02, 9.933365446728464e-04, 1.138945814827269e-05, 5.017342551847931e-05, 5.367146413563360e-02, 1.157339497221973e-04, 1.157339497221971e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_tw1_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tw1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [7.780081461536726e-07, 7.780050191335995e-07, 7.779836826618559e-07, 7.780303586991477e-07, 7.780069587333131e-07, 7.780069587333131e-07, 1.705788911421990e-04, 1.705846493110155e-04, 1.707051970245319e-04, 1.703984938817134e-04, 1.705734591236386e-04, 1.705734591236386e-04, 2.087726615031902e-02, 2.086617466568534e-02, 2.032118611310302e-02, 1.993275283409216e-02, 2.010256262304051e-02, 2.010256262304051e-02, 5.425314909569854e-01, 5.500132930289453e-01, 1.362443287476570e-02, 5.454024862698788e-01, 5.838529987338851e-01, 5.838529987338851e-01, 7.035496617297783e-02, 7.823398486729582e-02, 2.412262898808843e-01, 3.692215880218967e-02, 5.894509612925256e-02, 5.894509612925270e-02, 5.592740273140399e-05, 5.594528655432648e-05, 5.592788049552077e-05, 5.594367402113862e-05, 5.593666325670979e-05, 5.593666325670979e-05, 7.331365135061494e-04, 7.244250486684512e-04, 7.282113944768417e-04, 7.206993390827096e-04, 7.305614901754339e-04, 7.305614901754339e-04, 3.651391276397227e-02, 3.242234452041745e-02, 4.498894757886821e-02, 4.586500350558211e-02, 3.544882537372691e-02, 3.544882537372691e-02, 4.349743252591627e-01, 3.347362200316250e-01, 4.434362340908803e-01, 1.190247610310991e-03, 6.106856747183608e-01, 6.106856747183608e-01, 3.753632139840962e-02, 4.248061447950802e-02, 2.039968722320482e-01, 4.402628851881349e-01, 1.143629873968738e-01, 1.143629873968736e-01, 4.233748588835191e-02, 4.178082946934707e-02, 4.197569002193569e-02, 4.213869040220687e-02, 4.205725386998391e-02, 4.205725386998391e-02, 4.639418089861411e-02, 5.326447933284617e-02, 5.168705611438833e-02, 4.987239005839488e-02, 5.089347191275640e-02, 5.089347191275640e-02, 2.800260011559376e-02, 2.492142103668661e-01, 2.056048908768656e-01, 1.438212755483395e-01, 1.779445768577132e-01, 1.779445768577132e-01, 6.769695128919748e-02, 2.106170995343884e-01, 2.859304964166074e-01, 1.777878303809747e-01, 6.186685008245333e-01, 6.186685008245340e-01, 8.002302153555794e-02, 5.572479663151000e-02, 5.660962257102000e-02, 5.881288903173254e-01, 1.336620171589213e-01, 1.336620171589215e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05