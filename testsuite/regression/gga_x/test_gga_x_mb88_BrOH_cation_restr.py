
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_mb88_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_mb88", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.019634223239617e+01, -2.019638789431521e+01, -2.019662538613463e+01, -2.019594531821035e+01, -2.019629675446276e+01, -2.019629675446276e+01, -3.310273611973851e+00, -3.310264623016439e+00, -3.310138077458515e+00, -3.310986905745371e+00, -3.310342631698101e+00, -3.310342631698101e+00, -6.486090721396666e-01, -6.481958723709507e-01, -6.392611154447631e-01, -6.441352492331933e-01, -6.427547027558579e-01, -6.427547027558579e-01, -1.852397486160951e-01, -1.866108426010224e-01, -7.531936073879384e-01, -1.574174218107500e-01, -1.667243443475307e-01, -1.667243443475307e-01, -6.257899052356360e-02, -6.240794595521203e-02, -9.704889402478462e-02, -5.842854822211658e-02, -5.759921139648760e-02, -5.759921139648756e-02, -4.898938217422645e+00, -4.899317712975853e+00, -4.898961529169240e+00, -4.899296463278923e+00, -4.899127348778729e+00, -4.899127348778729e+00, -1.924199994349054e+00, -1.935855753211187e+00, -1.919323020166590e+00, -1.929589385653494e+00, -1.933031483818503e+00, -1.933031483818503e+00, -5.566366211549137e-01, -5.952164282580030e-01, -5.165688418387993e-01, -5.281935120519216e-01, -5.643512325057568e-01, -5.643512325057568e-01, -1.340872467537906e-01, -1.978483874850721e-01, -1.285503259888577e-01, -1.791987173590126e+00, -1.381583916366152e-01, -1.381583916366152e-01, -5.297853784019887e-02, -5.625374489225617e-02, -3.661898022201899e-02, -1.066511210306305e-01, -4.430084066485913e-02, -4.430084066485914e-02, -5.483250616996904e-01, -5.472448543546012e-01, -5.476288907751244e-01, -5.479394591287188e-01, -5.477839734255366e-01, -5.477839734255366e-01, -5.328577123806377e-01, -4.756050371596755e-01, -4.915506651386927e-01, -5.075999898444986e-01, -4.992961559398650e-01, -4.992961559398650e-01, -6.243787224568772e-01, -2.366158071548294e-01, -2.713636109170957e-01, -3.378739200407673e-01, -3.013288931167553e-01, -3.013288931167553e-01, -4.383014433383893e-01, -9.788585356215244e-02, -1.050216820087034e-01, -3.217085588318355e-01, -1.125728045622622e-01, -1.125728045622622e-01, -6.943238413050902e-02, -3.310173475325393e-02, -4.291428295793966e-02, -1.101330642520507e-01, -4.166034883700725e-02, -4.166034883700723e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_mb88_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_mb88", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.597044306007252e+01, -2.597052784307611e+01, -2.597090730198353e+01, -2.596964584325286e+01, -2.597030646315038e+01, -2.597030646315038e+01, -4.202984777201304e+00, -4.203011945653516e+00, -4.203742887117213e+00, -4.203270071168723e+00, -4.203118668647383e+00, -4.203118668647383e+00, -7.997032089092077e-01, -7.985051341751570e-01, -7.688426370135106e-01, -7.756554237450268e-01, -7.745501835936731e-01, -7.745501835936731e-01, -1.908306288899154e-01, -1.948077712154997e-01, -9.357912821593652e-01, -1.342488650397522e-01, -1.566027270733422e-01, -1.566027270733421e-01, -1.415788267028259e-02, -1.426095560514200e-02, -3.160889717111594e-02, -1.206156108990802e-02, -1.246043841262972e-02, -1.246043841262972e-02, -6.358827871688288e+00, -6.360613839668042e+00, -6.358911566994135e+00, -6.360488220961825e+00, -6.359732434765595e+00, -6.359732434765595e+00, -2.301075114939834e+00, -2.320037238311316e+00, -2.283684811109744e+00, -2.300455061662414e+00, -2.319958309776911e+00, -2.319958309776911e+00, -7.112430865490411e-01, -7.839414564501088e-01, -6.564039558754224e-01, -6.926374299036903e-01, -7.241531498900278e-01, -7.241531498900278e-01, -8.366494402442214e-02, -1.827368234432215e-01, -7.629224937467387e-02, -2.361774134673257e+00, -1.046767953658492e-01, -1.046767953658492e-01, -1.059796395678182e-02, -1.164711722384497e-02, -7.821533809465816e-03, -4.862019995496764e-02, -9.454655422552832e-03, -9.454655422552818e-03, -7.279723342252779e-01, -7.211728816277412e-01, -7.235508596772620e-01, -7.255243200482957e-01, -7.245356515510423e-01, -7.245356515510423e-01, -7.090256138517644e-01, -5.876163644146593e-01, -6.223491260194064e-01, -6.569145634421756e-01, -6.393368145103039e-01, -6.393368145103039e-01, -8.213956284305809e-01, -2.398286155426549e-01, -2.980233484737833e-01, -4.127001660091801e-01, -3.515384722472604e-01, -3.515384722472604e-01, -5.405206353250971e-01, -3.091127994042648e-02, -3.921873977579532e-02, -4.020719352674877e-01, -6.479854902296574e-02, -6.479854902296575e-02, -1.641823458892388e-02, -5.889043966165334e-03, -8.369262728548228e-03, -6.001998367310471e-02, -8.863499986035601e-03, -8.863499986035573e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_mb88_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_mb88", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.425537520840269e-09, -2.425515178893868e-09, -2.425399939823869e-09, -2.425732680554632e-09, -2.425560586611491e-09, -2.425560586611491e-09, -3.372278036852201e-06, -3.372303582427234e-06, -3.372564134565590e-06, -3.369551518600537e-06, -3.371983513917279e-06, -3.371983513917279e-06, -2.322285891861720e-03, -2.329484309435395e-03, -2.504741084104110e-03, -2.427482806475192e-03, -2.447053081502674e-03, -2.447053081502674e-03, -4.164392775485218e-01, -3.980140266026617e-01, -1.271210089996377e-03, -1.026205124272582e+00, -7.134563399939027e-01, -7.134563399939029e-01, -1.023737085937330e+04, -8.982570037188383e+03, -5.863625537655913e+01, -4.810348694436971e+04, -2.671149510873835e+04, -2.671149510873838e+04, -7.001678818444755e-07, -6.999566988815525e-07, -7.001547765510794e-07, -6.999683815948225e-07, -7.000625919625683e-07, -7.000625919625683e-07, -3.065012876194001e-05, -2.986688147985113e-05, -3.109044063440405e-05, -3.038452460024446e-05, -3.000800835622912e-05, -3.000800835622912e-05, -4.209416218681408e-03, -3.226718218276823e-03, -5.685162506678818e-03, -5.191833241351923e-03, -3.979908437306868e-03, -3.979908437306868e-03, -3.204279152375472e+00, -3.679801090522177e-01, -4.146695940451941e+00, -3.929315089876263e-05, -2.064237928741576e+00, -2.064237928741576e+00, -1.096035884349914e+05, -5.375003545665887e+04, -1.950252389823341e+05, -1.488173806394351e+01, -9.207216367153267e+04, -9.207216367153262e+04, -4.516890709776700e-03, -4.517502238331733e-03, -4.513981390102706e-03, -4.513552389878450e-03, -4.513477175243311e-03, -4.513477175243311e-03, -5.085797474563456e-03, -8.022326150933410e-03, -6.943314840251261e-03, -6.074680087269527e-03, -6.498899546869783e-03, -6.498899546869783e-03, -2.662957401624704e-03, -1.595417581074667e-01, -8.405556144912849e-02, -3.173352151383915e-02, -5.205557876365124e-02, -5.205557876365124e-02, -1.113518045212743e-02, -6.476888552214108e+01, -2.651477950707666e+01, -3.807002111214347e-02, -7.456757193124279e+00, -7.456757193124280e+00, -3.597748594187364e+03, -3.761904539667416e+06, -3.614803552365365e+05, -9.024715325049808e+00, -1.228958554261633e+05, -1.228958554261637e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05